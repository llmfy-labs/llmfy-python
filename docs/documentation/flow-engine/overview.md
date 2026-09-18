---
title: Overview
description: State-based workflow orchestrator for building LLM-powered pipelines.
---

# FlowEngine

FlowEngine is a state-based workflow orchestrator for building LLM-powered pipelines. It lets you define a graph of nodes (processing steps) connected by edges (transitions), manage shared state across nodes, and optionally persist state across sessions with checkpointers.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **State** | A `TypedDict` shared across all nodes. Each node reads from state and returns updates. |
| **Node** | A Python function (sync or async) that processes state and returns a dict of updates. |
| **Edge** | A connection between two nodes. Can be direct or conditional. |
| **START / END** | Special constants marking the entry and exit of the workflow. |
| **Checkpointer** | Optional backend that saves state after each node so sessions can resume. |

## Installation

FlowEngine state requires the `typing_extensions` package:

=== "UV"
    
    ```shell
    uv add "llmfy[typing_extensions]"
    ```

=== "pip"
    
    ```shell
    pip install "llmfy[typing_extensions]"
    ```

## Quick Start

```python linenums="1"
import asyncio
from typing import Annotated
from typing_extensions import TypedDict
from llmfy import FlowEngine, START, END


def add_messages(old: list, new: list) -> list:
    if old is None:
        return new
    return old + new


class AppState(TypedDict):
    messages: Annotated[list[str], add_messages]
    status: str


async def step_one(state: AppState) -> dict:
    return {"messages": ["step one done"], "status": "step1"}


async def step_two(state: AppState) -> dict:
    return {"messages": ["step two done"], "status": "step2"}


async def main():
    flow = FlowEngine(AppState)

    flow.add_node("step_one", step_one)
    flow.add_node("step_two", step_two)

    flow.add_edge(START, "step_one")
    flow.add_edge("step_one", "step_two")
    flow.add_edge("step_two", END)

    flow.build()

    result = await flow.invoke({"messages": [], "status": "start"})
    print(result)


asyncio.run(main())
```

## FlowEngine API

```python
from llmfy import FlowEngine
```

| Method | Description |
|--------|-------------|
| `FlowEngine(state_schema, checkpointer=None, max_steps=100, types=None, hooks=None)` | Create engine with a TypedDict state schema, optional checkpointer, step-limit guard, custom types for checkpoint (de)serialization, and observability hooks |
| `register_type(cls)` | Register a custom type (Pydantic `BaseModel` or `@dataclass`) that appears in state and must cross a Redis/SQL checkpointer's serialization boundary — equivalent to passing it in `types=[...]` at construction |
| `add_node(name, func, stream=False, retry=None, timeout=None)` | Add a processing node. Set `stream=True` for generator nodes; `retry` takes a `RetryPolicy`, `timeout` a per-attempt seconds limit |
| `add_edge(source, target)` | Add a direct transition. `target` can be a list (or `add_edge` called more than once from the same source) for static fan-out |
| `add_conditional_edges(source, condition_func, targets)` | Add conditional routing: `condition_func(state)` returns a target name (list form), a dict key (dict form), or a `Send`/`list[Send]` for dynamic fan-out |
| `build()` | Validate and compile the workflow. Returns the built `FlowEngine` |
| `invoke(apply_state=None, session_id=None, max_steps=None)` | Run the workflow synchronously. Returns the final state dict |
| `stream(apply_state=None, session_id=None, max_steps=None)` | Run the workflow with streaming. Returns an async generator of `FlowEngineStreamResponse` |
| `get_state(session_id)` | Retrieve the latest checkpointed state for a session |
| `get_checkpoint(session_id, checkpoint_id=None)` | Retrieve one specific checkpoint, or the latest if `checkpoint_id` is omitted |
| `list_checkpoints(session_id, limit=10)` | List checkpoint metadata for a session, newest first |
| `delete_checkpoints(session_id, checkpoint_id=None)` | Delete one checkpoint, or every checkpoint for the session if `checkpoint_id` is omitted |
| `reset_session(session_id)` | Clear all checkpoints for a session (start fresh) |
| `details()` | Print a text representation of the workflow graph |
| `visualize()` | Return a Mermaid diagram URL for the workflow |

See [Nodes & Edges](nodes-edges.md) for static/dynamic fan-out, retry/timeout, and hooks; [Checkpointer](checkpointer.md) for `types`/`register_type`, retention, and compression/encryption.

## Visualization

After calling `build()`, inspect or visualize the workflow:

```python linenums="1"
flow.build()

# Text representation
print(flow.details())

# Mermaid diagram URL
print(flow.visualize())
```
