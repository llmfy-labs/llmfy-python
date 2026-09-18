"""
Tool-calling LLM agent built on FlowEngine, driven with `stream()`. Both
nodes are `stream=True` async generators yielding `NodeStreamResponse` —
partial content streams to the terminal as it arrives, and the final
`NodeStreamResponse` (type=RESULT) carries the state update. When the
model's `thinking`/`reasoning` config is enabled (see
docs/documentation/features/thinking-config.md), its reasoning trace
streams too, wrapped in a `ThinkingChunk` so the terminal can tell it apart
from the model's actual answer — same `[thinking]...[/thinking]` framing as
thinking_example.py, just routed through a FlowEngine node instead of
`LLMfy` directly.

Requires a Google AI API key (`GOOGLE_API_KEY` in the environment / a
`.env` file) — see flowengine_agent_example.py for the non-streaming
equivalent.
"""

import asyncio
from dataclasses import dataclass
from typing import Annotated, TypedDict

from dotenv import load_dotenv

from llmfy import (
    END,
    START,
    # AnthropicMessagesConfig,
    # AnthropicMessagesModel,
    # AnthropicMessagesThinkingConfig,
    FlowEngine,
    GenerationResponse,
    # GoogleAIGenerateConfig,
    # GoogleAIGenerateModel,
    InMemoryCheckpointer,
    LLMfy,
    Message,
    NodeStreamResponse,
    NodeStreamType,
    OpenAIChatConfig,
    OpenAIChatModel,
    # OpenAIResponsesConfig,
    # OpenAIResponsesModel,
    # OpenAIResponsesReasoningConfig,
    Role,
    Tool,
    ToolNodeStreamResponse,
    ToolNodeStreamType,
    ToolRegistry,
    tools_stream_node,
)
from llmfy.flow_engine.stream.flow_engine_stream_response import FlowEngineStreamType

load_dotenv()


def append_messages(old: list[Message] | None, new: list[Message]) -> list[Message]:
    return (old or []) + new


@dataclass
class ThinkingChunk:
    """Marks a streamed `NodeStreamResponse.content` as reasoning/thinking
    text rather than the model's answer, the same way `ToolNodeStreamResponse`
    marks one as a tool-execution event — so `main()`'s terminal consumer can
    tell the three apart via `isinstance` and print each differently."""

    content: str


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
    #     model="gemini-2.5-flash-lite",
    #     config=GoogleAIGenerateConfig(temperature=0.7),
    # )
    # model = OpenAIResponsesModel(
    #     model="gpt-5",
    #     config=OpenAIResponsesConfig(
    #         temperature=None,
    #         reasoning=OpenAIResponsesReasoningConfig(enabled=True, effort="medium", summary="auto")
    #     ),
    # )
    # model = AnthropicMessagesModel(
    #     model="claude-haiku-4-5",
    #     config=AnthropicMessagesConfig(
    #         thinking=AnthropicMessagesThinkingConfig(enabled=True, budget_tokens=1024)
    #     ),
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

    async def main_orchestrator(state: AppState):
        full_content = ""
        last_message: Message | None = None

        async for chunk in llm.achat_stream(state["messages"]):
            if not isinstance(chunk, GenerationResponse):
                continue
            if chunk.messages:
                last_message = chunk.messages[-1]
            # Thinking is never persisted into conversation history (see
            # `AIResponse.thinking`), only streamed here for display — so it
            # doesn't touch `full_content`/`last_message` like real content
            # does below.
            if chunk.result.thinking:
                yield NodeStreamResponse(
                    type=NodeStreamType.STREAM,
                    content=ThinkingChunk(content=chunk.result.thinking),
                )
            if chunk.result.content:
                full_content += chunk.result.content
                yield NodeStreamResponse(
                    type=NodeStreamType.STREAM, content=chunk.result.content
                )

        yield NodeStreamResponse(
            type=NodeStreamType.RESULT,
            content=full_content,
            state={"messages": [last_message]},
        )

    async def tools_executor(state: AppState):
        for event in tools_stream_node(
            messages=state["messages"], registry=tool_registry
        ):
            if event.type == ToolNodeStreamType.EXECUTING:
                yield NodeStreamResponse(type=NodeStreamType.STREAM, content=event)
            elif event.type == ToolNodeStreamType.RESULT:
                yield NodeStreamResponse(
                    type=NodeStreamType.RESULT,
                    content=event,
                    state={"messages": [event.result]},
                )

    def should_continue(state: AppState) -> str:
        last_message = state["messages"][-1]
        return "tools" if last_message.tool_calls else END

    flow = FlowEngine(AppState, checkpointer=InMemoryCheckpointer())
    flow.add_node("main", main_orchestrator, stream=True)
    flow.add_node("tools", tools_executor, stream=True)
    flow.add_edge(START, "main")
    flow.add_edge("tools", "main")
    flow.add_conditional_edges("main", should_continue, ["tools", END])

    return flow.build()


async def main():
    agent = build_agent()
    session_id = "flowengine-agent-stream-example"

    print("=== FlowEngine agent (stream) — type 'exit' to quit ===\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        print("Assistant: ", end="", flush=True)
        is_thinking = False
        async for response in agent.stream(
            apply_state={"messages": [Message(role=Role.USER, content=user_input)]},
            session_id=session_id,
        ):
            if response.type == FlowEngineStreamType.STREAM:
                if isinstance(response.content, ThinkingChunk):
                    if not is_thinking:
                        print("\n-----[thinking]-----\n", end="", flush=True)
                        is_thinking = True
                    print(response.content.content, end="", flush=True)
                    continue
                if is_thinking:
                    print("\n-----[/thinking]-----\n\n", end="", flush=True)
                    is_thinking = False
                if isinstance(response.content, str):
                    print(response.content, end="", flush=True)
                elif isinstance(response.content, ToolNodeStreamResponse):
                    # `tools_executor` only ever streams EXECUTING events
                    # (type=STREAM) — its RESULT event is yielded as
                    # type=RESULT, which is this same `tools_stream_node`
                    # `event` but arrives below as the "tools" node's own
                    # FlowEngineStreamType.RESULT, never in this STREAM
                    # branch. See that branch for the RESULT case.
                    print(f"\n[calling tool: {response.content.name}]", flush=True)
            elif response.type == FlowEngineStreamType.RESULT:
                if (
                    isinstance(response.content, ToolNodeStreamResponse)
                    and response.content.type == ToolNodeStreamType.RESULT
                ):
                    print(
                        f"[tool result: {response.content.result.tool_results}]\n",  # type: ignore
                        flush=True,
                    )
        print("\n")


if __name__ == "__main__":
    asyncio.run(main())
