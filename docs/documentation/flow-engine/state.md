---
title: State
description: Define and manage workflow state using TypedDict and reducers in FlowEngine.
---

# State

State is a `TypedDict` shared across all nodes in a workflow. Each node receives the current state and returns a dict of updates.

## Defining State

```python linenums="1"
from typing import Annotated
from typing_extensions import TypedDict


class AppState(TypedDict):
    messages: list
    status: str
    counter: int
```

## Reducers

By default, a node's returned value **replaces** the existing field. Use `Annotated` to attach a **reducer function** — the reducer receives `(old_value, new_value)` and returns the merged result.

```python linenums="1"
from typing import Annotated
from typing_extensions import TypedDict


def add_messages(old: list, new: list) -> list:
    """Append new messages to existing list instead of replacing."""
    if old is None:
        return new
    return old + new


class AppState(TypedDict):
    messages: Annotated[list, add_messages]  # reducer: appends
    status: str                               # no reducer: replaces
    counter: int                              # no reducer: replaces
```

### Reducer vs Replace behaviour

Given a node that returns `{"messages": ["b"], "status": "done", "counter": 2}`:

| Field | Has Reducer | Previous | Returned | Result |
|-------|-------------|----------|----------|--------|
| `messages` | Yes (`add_messages`) | `["a"]` | `["b"]` | `["a", "b"]` |
| `status` | No | `"running"` | `"done"` | `"done"` |
| `counter` | No | `1` | `2` | `2` |

## Node State Updates

Node functions return a **partial dict** — only include keys that changed:

```python linenums="1"
async def my_node(state: AppState) -> dict:
    new_counter = state.get("counter", 0) + 1
    return {
        "counter": new_counter,
        "status": "processed",
        # "messages" not returned — it keeps its current value
    }
```

## State with Custom Objects

State fields can hold any Python object. `InMemoryCheckpointer` keeps live objects (deep-copied) and needs nothing extra. `RedisCheckpointer` and `SQLCheckpointer` cross a JSON serialization boundary, though, so a custom type (a Pydantic `BaseModel` or `@dataclass`) used in state must be **registered explicitly** — an unregistered type raises `CheckpointDeserializationException` rather than silently degrading:

```python linenums="1"
from typing import Annotated, List
from typing_extensions import TypedDict
from llmfy import FlowEngine, Message


def add_messages(old: List[Message], new: List[Message]) -> List[Message]:
    if old is None:
        return new
    return old + new


class ChatState(TypedDict):
    messages: Annotated[List[Message], add_messages]
    session_id: str


# Register at construction time...
flow = FlowEngine(ChatState, checkpointer=redis_checkpointer, types=[Message])
# ...or after the fact:
flow.register_type(Message)
```

## Continuation with Reducers

When using a checkpointer and the same `session_id`, the `apply_state` passed to `invoke` is **merged** into the checkpointed state via reducers before the workflow runs:

```python linenums="1"
# First run — starts fresh
state = await flow.invoke(
    {"messages": [], "status": "start", "counter": 0},
    session_id="user-123",
)

# Second run — merged with checkpoint via reducers
state = await flow.invoke(
    {"messages": ["new input"], "status": "continuing"},
    session_id="user-123",
)
# messages field: reducer appends "new input" to previous messages
# status field:   replaced with "continuing"
# counter field:  carried over from checkpoint unchanged

# Pass None to continue from checkpoint without any updates
state = await flow.invoke(None, session_id="user-123")
```
