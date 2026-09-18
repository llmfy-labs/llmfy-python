"""
Tool-calling LLM agent built on FlowEngine, driven with `invoke()`
(non-streaming). A conditional edge loops back to "tools" whenever the
model's last message requests a tool call, and to END otherwise.

Requires a Google AI API key (`GOOGLE_API_KEY` in the environment / a
`.env` file) — see flowengine_agent_stream_example.py for the streaming
equivalent.
"""

import asyncio
from typing import Annotated, TypedDict, cast

from dotenv import load_dotenv

from llmfy import (
    END,
    START,
    FlowEngine,
    # GoogleAIGenerateConfig,
    # GoogleAIGenerateModel,
    InMemoryCheckpointer,
    LLMfy,
    Message,
    OpenAIChatConfig,
    OpenAIChatModel,
    Role,
    Tool,
    ToolRegistry,
    tools_node,
)

load_dotenv()


def append_messages(old: list[Message] | None, new: list[Message]) -> list[Message]:
    return (old or []) + new


class AppState(TypedDict):
    messages: Annotated[list[Message], append_messages]


@Tool()
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is 22 degrees {unit}"


@Tool()
def get_current_time(location: str) -> str:
    """Get the current time for a location."""
    return f"The time in {location} is 09:00 AM"


def build_agent() -> FlowEngine:
    # model = GoogleAIGenerateModel(
    #     model="gemini-2.5-flash-lite", config=GoogleAIGenerateConfig(temperature=0.7)
    # )
    model = OpenAIChatModel(
        # model="gemma4:e4b",
        model="qwen3.5:9b",
        config=OpenAIChatConfig(temperature=0.7),
        api_key="ollama",  # unused by Ollama, but required by the openai SDK client
        base_url="http://localhost:11434/v1",  # Ollama proxy
    )

    llm = LLMfy(model, system_message="You are a helpful assistant.")

    tools = [get_current_weather, get_current_time]
    llm.register_tool(tools)
    tool_registry = ToolRegistry(tools, model)

    def main_orchestrator(state: AppState) -> dict:
        response = llm.chat(state["messages"])
        return {"messages": [response.messages[-1]]}

    def tools_executor(state: AppState) -> dict:
        results = tools_node(messages=state["messages"], registry=tool_registry)
        return {"messages": results}

    def should_continue(state: AppState) -> str:
        last_message = state["messages"][-1]
        return "tools" if last_message.tool_calls else END

    flow = FlowEngine(AppState, checkpointer=InMemoryCheckpointer())
    flow.add_node("main", main_orchestrator)
    flow.add_node("tools", tools_executor)
    flow.add_edge(START, "main")
    flow.add_edge("tools", "main")
    flow.add_conditional_edges("main", should_continue, ["tools", END])

    return flow.build()


async def main():
    agent = build_agent()
    print(agent.visualize())
    session_id = "flowengine-agent-example"

    print("=== FlowEngine agent (invoke) — type 'exit' to quit ===\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        result = await agent.invoke(
            apply_state={"messages": [Message(role=Role.USER, content=user_input)]},
            session_id=session_id,
        )
        reply = cast(Message, result["messages"][-1])
        print(f"Assistant: {reply.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
