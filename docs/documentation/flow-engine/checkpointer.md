---
title: Checkpointer
description: Persist and resume workflow state across sessions using InMemory, Redis, or SQL checkpointers.
---

# Checkpointer

A checkpointer persists workflow state after each node execution. When the same `session_id` is used again, the workflow resumes from the last saved state instead of starting fresh.

## Without a Checkpointer

Without a checkpointer, each `invoke` call starts from scratch — no state is saved between calls.

## With a Checkpointer

Pass a checkpointer to `FlowEngine`:

```python linenums="1"
from llmfy import FlowEngine

flow = FlowEngine(state_schema=AppState, checkpointer=checkpointer)
```

## Available Checkpointers

### InMemoryCheckpointer

Stores state in process memory. State is lost on restart. Best for development and testing.

```python linenums="1"
from llmfy.flow_engine.checkpointer.in_memory_checkpointer import InMemoryCheckpointer

checkpointer = InMemoryCheckpointer()
```

### RedisCheckpointer

Stores state in Redis. Persistent across restarts. Supports optional TTL.

!!! note "Requires"
    `pip install "llmfy[redis]"` and a running Redis instance.

```python linenums="1"
from llmfy.flow_engine.checkpointer.redis_checkpointer import RedisCheckpointer

checkpointer = RedisCheckpointer(
    redis_url="redis://localhost:6379/0",
    prefix="myapp:",    # key prefix in Redis
    ttl=3600,           # optional: expire after 1 hour (seconds)
)
```

### SQLCheckpointer

Stores state in a SQL database (PostgreSQL, MySQL, or SQLite). Auto-creates tables on first use.

!!! note "Requires"
    `pip install "llmfy[SQLAlchemy]"` plus a database driver.

    | Database | Async driver | Sync driver |
    |----------|-------------|-------------|
    | PostgreSQL | `asyncpg` | `psycopg2` |
    | MySQL | `aiomysql` | `pymysql` |
    | SQLite | `aiosqlite` | *(built-in)* |

```python linenums="1"
from llmfy.flow_engine.checkpointer.sql_checkpointer import SQLCheckpointer

# SQLite (simplest setup)
checkpointer = SQLCheckpointer(connection_string="sqlite:///checkpoints.db")

# MySQL (sync)
checkpointer = SQLCheckpointer(
    connection_string="mysql+pymysql://user:pass@localhost/dbname"
)

# PostgreSQL (async)
checkpointer = SQLCheckpointer(
    connection_string="postgresql+asyncpg://user:pass@localhost/dbname"
)
```

## Custom Types in State

`RedisCheckpointer`/`SQLCheckpointer` cross a JSON serialization boundary, so a Pydantic `BaseModel` or `@dataclass` used in state (e.g. `Message`) must be registered — otherwise a load raises `CheckpointDeserializationException`. `InMemoryCheckpointer` never needs this; it keeps live Python objects.

```python linenums="1"
flow = FlowEngine(AppState, checkpointer=redis_checkpointer, types=[MyCustomType])
# or, after construction:
flow.register_type(MyCustomType)
```

## Retention & Storage Options

All three backends support count-cap and/or TTL retention, combinable:

```python linenums="1"
checkpointer = InMemoryCheckpointer(max_checkpoints_per_session=20)
checkpointer = RedisCheckpointer(redis_url=..., ttl=3600, max_checkpoints_per_session=20)
checkpointer = SQLCheckpointer(connection_string=..., ttl_seconds=3600, max_checkpoints_per_session=20)
```

`max_checkpoints_per_session` keeps only the newest N checkpoints per session, pruned right after each save. `ttl`/`ttl_seconds` drops checkpoints older than a configured age (Redis's `ttl` is a sliding, whole-session expiry refreshed on every save — use `max_checkpoints_per_session` if you also want to prune old checkpoints out of an *active* session). Both are unbounded (`None`) by default.

`RedisCheckpointer`/`SQLCheckpointer` also accept `compress=True` (zlib, storage-size win) and `encryption_key=Fernet.generate_key()` (Fernet encryption, requires `pip install "llmfy[crypto]"`) — both off by default, compress-then-encrypt on write. `max_state_bytes` (default 25 MiB) rejects an oversized checkpoint with `CheckpointPayloadTooLargeException` before it's written.

## Session Continuation

### Automatic continuation

Passing the same `session_id` automatically continues from the last checkpoint. The `apply_state` is merged with the checkpointed state via reducers:

```python linenums="1"
# First invocation — starts fresh
result = await flow.invoke(
    {"messages": [], "status": "start", "counter": 0},
    session_id="user-123",
)

# Second invocation — continues from checkpoint
# "messages" reducer appends, others replace
result = await flow.invoke(
    {"messages": ["new input"], "status": "continuing"},
    session_id="user-123",
)

# Continue without any updates
result = await flow.invoke(None, session_id="user-123")
```

### New session

Use a different `session_id` to start a fresh workflow:

```python linenums="1"
result = await flow.invoke(
    {"messages": [], "status": "start", "counter": 0},
    session_id="user-456",   # new session, starts fresh
)
```

### Reset a session

Call `reset_session` to clear all checkpoints for a `session_id` and allow a fresh start:

```python linenums="1"
await flow.reset_session("user-123")

# Next invoke starts fresh even with the same session_id
result = await flow.invoke(
    {"messages": [], "status": "fresh"},
    session_id="user-123",
)
```

## Inspecting State

```python linenums="1"
# Get latest state for a session
state = await flow.get_state("user-123")

# List recent checkpoints
checkpoints = await flow.list_checkpoints("user-123", limit=5)

# Get one specific checkpoint, or the latest if checkpoint_id is omitted
checkpoint = await flow.get_checkpoint("user-123", checkpoint_id=None)

# Delete one checkpoint, or every checkpoint for the session
await flow.delete_checkpoints("user-123", checkpoint_id=None)

# Use state to decide whether to continue or restart
if state and state.get("status") == "done":
    await flow.reset_session("user-123")
    await flow.invoke({"messages": [], "status": "restart"}, session_id="user-123")
else:
    await flow.invoke(None, session_id="user-123")
```

Every `Checkpoint` carries `metadata` (`checkpoint_id`, `session_id`, `run_id`, `created_at`, `node`, `prev_node`, `step`, `updated_fields`, `attempt`, `dispatch_id`) — useful for tracing which nodes ran together in one `invoke()`/`stream()` call. `RedisCheckpointer` additionally has `await checkpointer.list_by_run(session_id, run_id, limit=100)` to fetch every checkpoint from one run cheaply, without walking the session's whole history.

## Resume Limitations

- Resuming right after a *conditional* node doesn't re-derive its routing decision from the checkpoint alone — the caller restarts from `START` instead.
- Resuming mid a *static* fan-out (multiple branches in flight when the process stopped) isn't supported; resume takes the first target of the completed node.
- A *dynamic* (`Send`) fan-out's atomic commit means this doesn't apply there — a checkpoint only ever exists for a fully-committed dispatch, never a partial one.
