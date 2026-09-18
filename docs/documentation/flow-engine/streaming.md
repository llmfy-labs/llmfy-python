---
title: Streaming
description: Stream LLM tokens and tool events from FlowEngine nodes in real time.
---

# Streaming

FlowEngine supports token-by-token streaming from LLM nodes. Stream nodes yield `NodeStreamResponse` objects and must be registered with `stream=True`.

## Stream Response Types

### FlowEngineStreamResponse

Top-level events emitted by `flow.stream()`:

```python
from llmfy.flow_engine.stream.flow_engine_stream_response import (
    FlowEngineStreamResponse,
    FlowEngineStreamType,
)
```

| Type | Description |
|------|-------------|
| `FlowEngineStreamType.START` | Workflow execution started |
| `FlowEngineStreamType.STREAM` | Intermediate chunk from a stream node |
| `FlowEngineStreamType.RESULT` | Node completed, `state` field contains updated state |
| `FlowEngineStreamType.ERROR` | An error occurred |

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `type` | `str` | Event type (`FlowEngineStreamType`) |
| `node` | `str` | Name of the node that produced the event |
| `content` | `Any` | Streamed content (string, `ToolNodeStreamResponse`, etc.) |
| `state` | `dict` | Snapshot after this event (see the dynamic fan-out caveat under [Hooks](nodes-edges.md#hooks): during a `Send` dispatch, a branch's own event sees state as of *before* the dispatch, not yet including its own update) |
| `error` | `Any` | Error details (only on `ERROR`) |
| `branch_index` / `branch_total` | `int \| None` | This branch's 0-based index and the dispatch's total branch count — set only for a dynamic (`Send`) fan-out branch, `None` otherwise |
| `dispatch_id` | `str \| None` | Shared by every branch of one `Send` dispatch, unique across dispatches — lets you tell "tool call 2 of 3" apart when every branch shares the same `node` name |

### NodeStreamResponse

Events yielded by stream node functions:

```python
from llmfy.flow_engine.stream.node_stream_response import (
    NodeStreamResponse,
    NodeStreamType,
)
```

| Type | `content` | `state` |
|------|-----------|---------|
| `NodeStreamType.STREAM` | Partial text or tool event | `None` |
| `NodeStreamType.RESULT` | Final content | Updated state dict |

## Writing a Stream Node

Stream node functions are **generators** that yield `NodeStreamResponse` objects. Register them with `stream=True`.

```python linenums="1"
from llmfy import GenerationResponse, LLMfy
from llmfy.flow_engine.stream.node_stream_response import NodeStreamResponse, NodeStreamType


def stream_node(state: AppState):
    response = NodeStreamResponse()

    stream = llm.chat_stream(state.get("messages", []))
    full_content = ""
    result_messages = []

    for chunk in stream:
        if isinstance(chunk, GenerationResponse):
            if chunk.messages:
                result_messages = chunk.messages
            if chunk.result.content:
                full_content += chunk.result.content

                # Yield intermediate chunk
                response.type = NodeStreamType.STREAM
                response.content = chunk.result.content
                response.state = None
                yield response

    # Yield final result with state update
    response.type = NodeStreamType.RESULT
    response.content = full_content
    response.state = {"messages": [result_messages[-1]]}
    yield response


# Register with stream=True
flow.add_node("main", stream_node, stream=True)
```

## Streaming Tool Execution

Use `tools_stream_node` to stream tool events:

```python linenums="1"
from llmfy.flow_engine.helper.tools_node.tools_node import tools_stream_node
from llmfy.flow_engine.stream.tool_node_stream_response import (
    ToolNodeStreamResponse,
    ToolNodeStreamType,
)


def tools_executor(state: AppState):
    response = NodeStreamResponse()

    for tool in tools_stream_node(state.get("messages", []), registry=tool_registry):
        if isinstance(tool, ToolNodeStreamResponse):
            if tool.type == ToolNodeStreamType.EXECUTING:
                # Notify that a tool is about to run
                response.type = NodeStreamType.STREAM
                response.content = tool
                response.state = None
                yield response

            if tool.type == ToolNodeStreamType.RESULT:
                # Tool finished — update state with result
                response.type = NodeStreamType.RESULT
                response.content = tool
                response.state = {"messages": [tool.result]}
                yield response


flow.add_node("tools", tools_executor, stream=True)
```

## Consuming the Stream

Call `flow.stream()` and iterate with `async for`:

```python linenums="1"
import asyncio
from llmfy import Message
from llmfy.flow_engine.stream.flow_engine_stream_response import FlowEngineStreamType
from llmfy.flow_engine.stream.tool_node_stream_response import ToolNodeStreamResponse
from llmfy.llmfy_core.messages.role import Role


async def chat(message: str):
    stream = agent.stream(
        {"messages": [Message(role=Role.USER, content=message)]},
        session_id="session-1",
    )

    async for chunk in stream:
        if chunk.type == FlowEngineStreamType.STREAM:
            if isinstance(chunk.content, str):
                # LLM token
                print(chunk.content, end="", flush=True)
            elif isinstance(chunk.content, ToolNodeStreamResponse):
                # Tool executing notification
                print(f"\nExecuting tool: {chunk.content.name}...")

        elif chunk.type == FlowEngineStreamType.RESULT:
            if isinstance(chunk.content, ToolNodeStreamResponse):
                tool = chunk.content
                if hasattr(tool.result, "tool_results"):
                    print(f"Tool result: {tool.result.tool_results}\n")

    print()  # newline after stream


asyncio.run(chat("What is the weather in London?"))
```
