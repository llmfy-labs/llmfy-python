---
title: Create Agent
description: Build an LLM agent loop with tool calling using FlowEngine.
---

# Create Agent

The most common FlowEngine pattern is an **agent loop**: the LLM runs, optionally calls tools, and loops back until it produces a final text response.

## Setup

```python linenums="1"
from llmfy import (
    LLMfy, BedrockConverseModel, BedrockConverseConfig,
    Message, Tool, ToolRegistry, tools_node,
    FlowEngine, InMemoryCheckpointer, START, END,
)
from typing import List, Annotated
from typing_extensions import TypedDict
```

## State Schema

Use a message list with a reducer so messages accumulate across nodes:

```python linenums="1"
def add_messages(old: List[Message], new: List[Message]) -> List[Message]:
    if old is None:
        return new
    return old + new


class AppState(TypedDict):
    messages: Annotated[List[Message], add_messages]
    status: str
```

## Define the Agent

```python linenums="1"
import asyncio
from llmfy.llmfy_core.messages.role import Role

# 1. Model & LLM
model = BedrockConverseModel(model="amazon.nova-lite-v1:0", config=BedrockConverseConfig(temperature=0.7))
llm = LLMfy(model, system_message="You are a helpful assistant.")

# 2. Tools
@Tool()
def get_current_weather(location: str, unit: str = "celsius") -> str:
    return f"The weather in {location} is 22 degrees {unit}"

@Tool()
def get_current_time(location: str) -> str:
    return f"The time in {location} is 09:00 AM"

tools = [get_current_weather, get_current_time]
llm.register_tool(tools)
tool_registry = ToolRegistry(tools, model)

# 3. Nodes
def main_orchestrator(state: AppState) -> dict:
    messages = state.get("messages", [])
    response = llm.chat(messages)
    ai_message = response.messages[-1]
    return {"messages": [ai_message], "status": "main"}


def tools_executor(state: AppState) -> dict:
    results = tools_node(
        messages=state.get("messages", []),
        registry=tool_registry,
    )
    return {"messages": results}


def should_continue(state: AppState) -> str:
    last_message = state.get("messages", [])[-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 4. Build flow
flow = FlowEngine(state_schema=AppState, checkpointer=InMemoryCheckpointer())

flow.add_node("main", main_orchestrator)
flow.add_node("tools", tools_executor)

flow.add_edge(START, "main")
flow.add_edge("tools", "main")
flow.add_conditional_edges("main", should_continue, ["tools", END])

agent = flow.build()

# 5. Invoke
async def chat(message: str) -> Message:
    result = await agent.invoke(
        {"messages": [Message(role=Role.USER, content=message)]},
        session_id="session-1",
    )
    return result["messages"][-1]


async def main():
    reply = await chat("What is the weather and time in London?")
    print(f"Assistant: {reply.content}")


asyncio.run(main())
```

## How the Agent Loop Works

```
START
  │
  ▼
main ──(tool_calls?)──► tools
  ▲                        │
  └────────────────────────┘
  │
  ▼ (no tool_calls)
 END
```

1. `main` sends messages to the LLM
2. If the LLM returns tool calls → route to `tools`
3. `tools` executes them and appends results to messages
4. Route back to `main` — the LLM processes tool results and responds
5. When no more tool calls → route to `END`

## Multi-turn Chat with Session

Pass the same `session_id` across calls to maintain conversation history:

```python linenums="1"
async def main():
    # Turn 1
    r1 = await agent.invoke(
        {"messages": [Message(role=Role.USER, content="Hello!")]},
        session_id="user-abc",
    )
    print(r1["messages"][-1].content)

    # Turn 2 — continues from checkpointed state
    r2 = await agent.invoke(
        {"messages": [Message(role=Role.USER, content="What is the weather in Paris?")]},
        session_id="user-abc",
    )
    print(r2["messages"][-1].content)
```
