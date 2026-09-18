# FlowEngine Agent Basic Example

Examples demonstrating FlowEngine with normal node (non-streaming node) and run with invoke.

```python
import asyncio
import os
from typing import List, TypedDict, cast

from dotenv import load_dotenv
from sqlalchemy.engine import URL
from typing_extensions import Annotated

from llmfy import (
    BedrockConverseConfig,
    BedrockConverseModel,
    LLMfy,
    Message,
    Role,
    Tool,
    ToolRegistry,
    tools_node,
    RedisCheckpointer,
    SQLCheckpointer,
    FlowEngine,
    END, 
    START,
)

load_dotenv()


db_url = URL.create(
    drivername="mysql+pymysql",
    username=os.getenv("MYSQL_USER", "root"),
    password=os.getenv("MYSQL_PASSWORD", ""),
    host=os.getenv("MYSQL_HOST", "localhost"),
    port=int(os.getenv("MYSQL_PORT", 3306)),
    database=os.getenv("MYSQL_DATABASE", ""),
    query={"charset": "utf8mb4"},
)


def add_message(old_messages: List[Message], new_message: List[Message]):
    """Reducer function to append messages."""
    if old_messages is None:
        return new_message
    return old_messages + new_message


class AppState(TypedDict):
    messages: Annotated[list[Message], add_message]
    status: str


def build_agent(use_redis: bool = True):
    print("\n" + "=" * 60)
    print("Example Agent: Complex State with Custom Objects")
    print("=" * 60 + "\n")

    if use_redis:
        checkpointer = RedisCheckpointer(
            redis_url="redis://localhost:6379/0",
            prefix="flowengine:",
            ttl=3600,  # 1 hour TTL
        )
    else:
        checkpointer = SQLCheckpointer(
            connection_string=db_url.render_as_string(hide_password=False),
            echo=False,  # Set to True to see SQL queries
        )

    # checkpointer = MemoryCheckpointer()

    # Define a sample tool
    @Tool()
    def get_current_weather(location: str, unit: str = "celsius") -> str:
        return f"The weather in {location} is 22 degrees {unit}"

    @Tool()
    def get_current_time(location: str) -> str:
        return f"The time in {location} is 09:00 AM"

    model = BedrockConverseModel(
        # model="amazon.nova-pro-v1:0",
        # model="amazon.nova-pro-v1:0",
        # model="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        # model="anthropic.claude-3-haiku-20240307-v1:0",
        # model="us.meta.llama3-3-70b-instruct-v1:0",
        model="amazon.nova-lite-v1:0",
        config=BedrockConverseConfig(temperature=0.7),
    )

    # model = OpenAIChatModel(model="gpt-4o-mini", config=OpenAIChatConfig())

    llm = LLMfy(model, system_message="You are Hoki a helpfull assistant.")

    tools = [get_current_weather, get_current_time]

    # Register tool
    llm.register_tool(tools)

    # Register to ToolRegistry
    tool_registry = ToolRegistry(tools, model)

    flow = FlowEngine(state_schema=AppState, checkpointer=checkpointer)

    def main_orchestrator(state: AppState):
        messages = state.get("messages", [])
        # for msg in messages:
        #     print(f"- {msg}")
        response = llm.chat(messages)
        ai_response = response.messages[-1]

        return {"messages": [ai_response], "status": "main"}

    def tools_executor(state):
        tool_results = tools_node(
            messages=state.get("messages", []),
            registry=tool_registry,
        )
        return {"messages": tool_results}

    def should_continue(state):
        messages = state.get("messages", [])
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    flow.add_node("main", main_orchestrator)
    flow.add_node("tools", tools_executor)

    flow.add_edge(START, "main")
    flow.add_edge("tools", "main")
    flow.add_conditional_edges("main", should_continue, ["tools", END])

    return flow.build()


# ============================================================================
# Main execution
# ============================================================================

agent = build_agent(use_redis=False)


async def chat(message: str):
    result = await agent.invoke(
        {
            "messages": [Message(role=Role.USER, content=message)],
        },
        session_id="cobalagi",
    )
    return cast(Message, result["messages"][-1])


async def main():
    print("=== Terminal Chat ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_msg = input("You: ")

        if user_msg.strip().lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye! 👋")
            break

        reply = await chat(user_msg)
        print(f"Chatbot: {reply.content}")


if __name__ == "__main__":
    asyncio.run(main())

```