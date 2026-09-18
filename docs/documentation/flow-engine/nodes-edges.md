---
title: Nodes & Edges
description: Define nodes and edges including direct, conditional, and loop patterns in FlowEngine.
---

# Nodes & Edges

## START and END

`START` and `END` are special constants that mark the entry and exit points of every workflow.

```python
from llmfy import START, END
```

Every workflow must have exactly one edge from `START` and at least one edge to `END`.

## Nodes

A node is any Python function (sync or async) that takes the current state and returns a dict of updates.

```python linenums="1"
from llmfy import FlowEngine, START, END

flow = FlowEngine(AppState)

async def my_node(state: AppState) -> dict:
    return {"status": "done"}

flow.add_node("my_node", my_node)
```

### Async vs Sync

Both async and sync node functions are supported:

```python linenums="1"
# Async node
async def async_node(state: AppState) -> dict:
    return {"status": "async"}

# Sync node
def sync_node(state: AppState) -> dict:
    return {"status": "sync"}
```

## Direct Edges

A direct edge routes unconditionally from one node to another:

```python linenums="1"
flow.add_edge(START, "node_a")
flow.add_edge("node_a", "node_b")
flow.add_edge("node_b", END)
```

This creates a simple linear workflow: `START → node_a → node_b → END`.

## Conditional Edges

A conditional edge routes to one of several nodes based on the current state. The condition function receives the state and returns the name of the next node (or `END`).

```python linenums="1"
def route(state: AppState) -> str:
    if state.get("counter", 0) < 5:
        return "low_path"
    return "high_path"

flow.add_conditional_edges(
    "check",                     # source node
    route,                       # condition function
    ["low_path", "high_path"],   # all possible targets
)
```

`targets` can also be a dict, mapping whatever label `route` returns to the real target node — handy when the condition function's return value reads better as a label than a node name:

```python linenums="1"
def check_result(state: AppState) -> str:
    return "Pass" if state["ok"] else "Fail"

flow.add_conditional_edges(
    "generate", check_result, {"Pass": END, "Fail": "improve"}
)
```

A node becomes a *conditional* node automatically the moment it's the source of a conditional edge.

## Example: Linear Workflow

```python linenums="1"
import asyncio
from typing_extensions import TypedDict
from llmfy import FlowEngine, START, END


class AppState(TypedDict):
    result: str
    status: str


async def process(state: AppState) -> dict:
    return {"result": "processed", "status": "done"}


async def main():
    flow = FlowEngine(AppState)

    flow.add_node("process", process)
    flow.add_edge(START, "process")
    flow.add_edge("process", END)
    flow.build()

    result = await flow.invoke({"result": "", "status": "start"})
    print(result)


asyncio.run(main())
```

## Example: Conditional Workflow

```python linenums="1"
import asyncio
from typing import Annotated
from typing_extensions import TypedDict
from llmfy import FlowEngine, START, END


def add_messages(old, new):
    return (old or []) + new


class AppState(TypedDict):
    messages: Annotated[list, add_messages]
    counter: int
    status: str


async def check(state: AppState) -> dict:
    return {"status": "checked"}


async def low_path(state: AppState) -> dict:
    return {"messages": ["low"], "counter": state["counter"] + 1, "status": "low"}


async def high_path(state: AppState) -> dict:
    return {"messages": ["high"], "counter": state["counter"] + 10, "status": "high"}


def route(state: AppState) -> str:
    if state.get("counter", 0) < 5:
        return "low_path"
    return "high_path"


async def main():
    flow = FlowEngine(AppState)

    flow.add_node("check", check)
    flow.add_node("low_path", low_path)
    flow.add_node("high_path", high_path)

    flow.add_edge(START, "check")
    flow.add_conditional_edges("check", route, ["low_path", "high_path"])
    flow.add_edge("low_path", END)
    flow.add_edge("high_path", END)

    flow.build()

    result = await flow.invoke({"messages": [], "counter": 2, "status": "start"})
    print(result)  # takes low_path (counter=2 < 5)


asyncio.run(main())
```

## Loop Pattern

Route back to a previous node to create a loop, and use `END` in the condition to break out:

```python linenums="1"
def should_loop(state: AppState) -> str:
    if state.get("counter", 0) < 3:
        return "main_node"   # loop back
    return END               # exit

flow.add_node("main_node", main_node)
flow.add_edge(START, "main_node")
flow.add_conditional_edges("main_node", should_loop, ["main_node", END])
flow.build()
```

## Static Fan-out / Fan-in

Passing a `list` as `target` (or calling `add_edge` more than once from the same source) fans out to a **fixed, build-time-known** set of branches — every target runs **concurrently**:

```python linenums="1"
flow.add_edge("fetch", ["research_facts", "research_opinions"])
flow.add_edge("research_facts", "combine")
flow.add_edge("research_opinions", "combine")
flow.add_edge("combine", END)
```

`combine` has two predecessors, so it becomes a *join*: the engine waits for both branches before running it exactly once, regardless of completion order. Use an `Annotated[Type, reducer_fn]` field (see [State](state.md)) so each branch's update merges safely instead of clobbering the other's.

## Dynamic Fan-out (`Send`)

When the number of branches is only known **at runtime** — e.g. one branch per item in a list whose length depends on the model's output — a conditional edge's routing function can return a `Send`, or a `list[Send]`, instead of a plain node name:

```python linenums="1"
from llmfy import Send

def route_to_joke_per_topic(state: JokeState) -> list[Send]:
    return [Send("tell_joke", {"topic": t}) for t in state["topics"]]

async def tell_joke(state: dict) -> dict:
    # `state` here is exactly {"topic": ...} — Send.state REPLACES the
    # branch's input, it does not merge with the router's shared state.
    return {"jokes": [make_joke(state["topic"])]}

flow.add_conditional_edges("plan_topics", route_to_joke_per_topic, ["tell_joke"])
```

Key rules:

- Every `Send` from one routing call must target the **same** declared node.
- `Send.state` fully replaces that branch's input; the branch's **returned** updates still merge into shared state normally.
- **Atomic commit**: every branch's update lands together, only once *every* branch succeeds — if any branch raises, the whole dispatch commits nothing (not even branches that finished first), and exactly one checkpoint is saved for the whole dispatch.
- Every event/hook for a dispatch also carries `branch_index`, `branch_total`, and `dispatch_id` (see [Streaming](streaming.md) and [Hooks](#hooks)), since all branches of one dispatch share the same node name.
- Sizing `max_steps` for a dispatch of N branches needs roughly `N + 2` steps (router + N branches + reduce node), not just `N`.

See `llmfy/example/flowengine_dynamic_fan_out_example.py` for a full runnable example.

## Retry & Timeout

```python linenums="1"
from llmfy import RetryPolicy

flow.add_node(
    "flaky",
    flaky_node,
    retry=RetryPolicy(max_attempts=3, backoff_seconds=0.05, retry_on=(FlakyServiceError,)),
)
flow.add_node("slow", slow_node, timeout=0.05)  # seconds, per attempt
```

`RetryPolicy(max_attempts=1, backoff_seconds=0.0, backoff_multiplier=2.0, retry_on=(Exception,))` — `max_attempts=1` (the default) means no retry. Backoff is exponential: `backoff_seconds * backoff_multiplier ** (attempt - 1)`. On exhaustion, a failure raises `NodeExecutionException` (or `NodeTimeoutException` for a timed-out attempt) rather than the node's raw exception. Under a dynamic fan-out, retry/timeout apply **per branch** independently.

## Hooks

Pass `FlowEngineHooks` when constructing the engine to observe node lifecycle events (tracing/logging) without touching node code:

```python linenums="1"
from llmfy import FlowEngine, FlowEngineHooks

flow = FlowEngine(
    AppState,
    hooks=FlowEngineHooks(
        on_node_start=lambda name, state: ...,            # (node_name, state)
        on_node_end=lambda name, state, updates: ...,      # (node_name, state, updates)
        on_error=lambda name, exc: ...,                    # (node_name, exception)
        on_branch_start=lambda name, idx, total, state: ...,         # Send branches only
        on_branch_end=lambda name, idx, total, state, updates: ...,  # Send branches only
    ),
)
```

`on_node_start`/`on_node_end`/`on_error` fire for every node, including each branch of a dynamic fan-out. `on_branch_start`/`on_branch_end` fire *in addition*, only for `Send` branches — use them to tell branches of one dispatch apart (their `node_name` is identical for all of them). Any callback may be sync or async. Hooks are set once per `FlowEngine` instance, not per call.
