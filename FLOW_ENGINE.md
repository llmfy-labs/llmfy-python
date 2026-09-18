# FlowEngine

`FlowEngine` (`llmfy/flow_engine/`) is `llmfy`'s graph-based workflow orchestrator — independent of the `LLMfy` chat API. You define nodes connected by edges, run them from `START` to `END`, and let a shared `TypedDict` state flow through the graph.

This document covers the engine as it stands after its full breaking rewrite: fixed concurrency semantics, static and dynamic fan-out, retry/timeout, lifecycle hooks, and safe checkpointing. Runnable versions of every example below live under [`llmfy/example/flowengine_*.py`](llmfy/example/).

## Contents

- [FlowEngine](#flowengine)
  - [Contents](#contents)
  - [Core concepts](#core-concepts)
  - [Quick start](#quick-start)
  - [State \& reducers](#state--reducers)
  - [Nodes](#nodes)
  - [Edges](#edges)
  - [Conditional routing](#conditional-routing)
  - [Static fan-out / fan-in](#static-fan-out--fan-in)
  - [Dynamic fan-out (`Send`)](#dynamic-fan-out-send)
  - [Retry \& timeout](#retry--timeout)
  - [Hooks](#hooks)
  - [Tool-calling helpers](#tool-calling-helpers)
  - [Checkpointing \& resume](#checkpointing--resume)
  - [Streaming](#streaming)
  - [Visualizing a workflow](#visualizing-a-workflow)
  - [Exceptions](#exceptions)
  - [Further examples](#further-examples)
  - [DEBUG manual on redis](#debug-manual-on-redis)
    - [by run id](#by-run-id)

## Core concepts

| Concept | Description |
|---|---|
| **State** | A `TypedDict` shared across all nodes. Each node reads it and returns a `dict` of updates. |
| **Node** | A sync/async function `(state) -> dict`, or an (async) generator yielding `NodeStreamResponse` for a streaming node. |
| **Edge** | A connection between nodes — plain (`add_edge`) or conditional (`add_conditional_edges`). |
| **START / END** | Reserved node names marking entry and exit. |
| **`ExecutionContext`** | Per-call state (session id, run id, state dict, step counter) — created fresh on every `invoke()`/`stream()` call, fresh or resumed, so two concurrent calls on the same `FlowEngine` never share mutable state and `max_steps` guards one call rather than a session's whole lifetime. |
| **Checkpointer** | Optional backend (`InMemoryCheckpointer`, `RedisCheckpointer`, `SQLCheckpointer`) that persists state so a session can resume. |

## Quick start

```python
import asyncio
from typing import Annotated, TypedDict
from llmfy import FlowEngine, START, END


def append_log(old: list[str] | None, new: list[str]) -> list[str]:
    return (old or []) + new


class AppState(TypedDict):
    log: Annotated[list[str], append_log]
    counter: int


async def start_node(state: AppState) -> dict:
    return {"log": ["started"], "counter": 0}


async def increment_node(state: AppState) -> dict:
    counter = state["counter"] + 1
    return {"log": [f"increment -> {counter}"], "counter": counter}


def keep_looping(state: AppState) -> str:
    return "continue" if state["counter"] < 3 else "stop"


async def main():
    flow = FlowEngine(AppState)
    flow.add_node("start", start_node)
    flow.add_node("increment", increment_node)

    flow.add_edge(START, "start")
    flow.add_edge("start", "increment")
    flow.add_conditional_edges(
        "increment", keep_looping, {"continue": "increment", "stop": END}
    )

    flow.build()  # validates the graph before anything can run

    result = await flow.invoke()
    print(result["counter"], result["log"])


asyncio.run(main())
```

`build()` validates structure before any execution is allowed: unreachable nodes only warn, but a missing START/END path, an undefined node reference, or an ambiguous (non-conditional) multi-target node raises `GraphValidationException`.

## State & reducers

State is a plain `TypedDict`. A field annotated `Annotated[Type, reducer_fn]` merges via `reducer(old_value, new_value)` on every update from a node; an unannotated field is replaced outright.

```python
def union_reducer(old: set | None, new: set) -> set:
    return (old or set()) | set(new)

class AppState(TypedDict):
    visited: Annotated[set, union_reducer]  # merged
    status: str                              # replaced
```

`reducer_fn` must take exactly two positional parameters — validated at `FlowEngine(...)` construction time.

## Nodes

```python
flow.add_node(
    name: str,
    func: Callable,           # sync or async: (state) -> dict
    stream: bool = False,     # see Streaming
    retry: RetryPolicy | None = None,
    timeout: float | None = None,
)
```

A node function returns a `dict` of updates (or `None`/`{}` for no update) — never the full state. `START`/`END` are reserved names and cannot be used as a node name.

## Edges

```python
flow.add_edge(source: str, target: str | list[str])
```

- `target` a single string: a plain transition.
- `target` a list, or `add_edge` called more than once from the same source: **static fan-out** (see below).
- `source` can be `START`; `target` can include `END`; `source == END` or `target == START` both raise `GraphValidationException`, as does an edge that targets itself.

## Conditional routing

```python
flow.add_conditional_edges(
    source: str,
    condition_func: Callable,       # (state) -> str | Send | list[Send]
    targets: list[str] | dict[str, str],
)
```

Two ways to declare `targets`:

**List form** — the routing function returns a target node name directly:

```python
def route(state) -> str:
    return "looper" if state["count"] < 3 else END

flow.add_conditional_edges("looper", route, ["looper", END])
```

**Dict form** (LangGraph-style) — the routing function returns a label, and the dict maps that label to the real target:

```python
def check_result(state) -> str:
    return "Pass" if state["ok"] else "Fail"

flow.add_conditional_edges(
    "generate", check_result, {"Pass": END, "Fail": "improve"}
)
```

A routing function that returns something not in `targets` (or not a key of the dict) raises `GraphValidationException`. A node becomes `CONDITIONAL` type automatically once it's the source of a conditional edge.

## Static fan-out / fan-in

Fan out to a fixed, build-time-known set of branches:

```python
flow.add_edge("fetch", ["research_facts", "research_opinions"])
flow.add_edge("research_facts", "combine")
flow.add_edge("research_opinions", "combine")
flow.add_edge("combine", END)
```

Both branches run **concurrently**. `combine` has two static predecessors, so it's a *join*: the engine waits for both branches to arrive before running it exactly once, regardless of completion order. Reducers are what let each branch's update merge safely (`visited: Annotated[set, union_reducer]` above, for instance).

Static fan-out's branch **count is fixed before `build()`** — if you need the branch count to depend on runtime data, use dynamic fan-out.

## Dynamic fan-out (`Send`)

A conditional edge's routing function can return a `Send`, or a `list[Send]`, to dispatch a **runtime-determined** number of parallel invocations of one declared target node — e.g. one branch per item in a list whose length is only known when the router runs (a "map" over runtime data).

```python
from llmfy import Send

class JokeState(TypedDict):
    topics: list[str]
    jokes: Annotated[list[str], lambda old, new: (old or []) + new]

async def plan_topics(state: JokeState) -> dict:
    return {"topics": fetch_topics_from_somewhere()}  # length only known now

def route_to_joke_per_topic(state: JokeState) -> list[Send]:
    return [Send("tell_joke", {"topic": t}) for t in state["topics"]]

async def tell_joke(state: dict) -> dict:
    # `state` is exactly {"topic": ...} — Send.state REPLACES the input,
    # it does not merge with plan_topics's shared state.
    return {"jokes": [make_joke(state["topic"])]}

async def finalize(state: JokeState) -> dict:
    # Runs exactly once, after every tell_joke branch has completed.
    ...

flow.add_edge(START, "plan_topics")
flow.add_conditional_edges("plan_topics", route_to_joke_per_topic, ["tell_joke"])
flow.add_edge("tell_joke", "finalize")
flow.add_edge("finalize", END)
```

Rules and guarantees:

- `Send.node` must be one of the conditional edge's declared `targets`; every `Send` from one routing call must target the **same** node (fanning out to different nodes in one call raises `GraphValidationException`).
- `Send.state` fully **replaces** that branch's input — the branch never sees the router's shared state, only what you put in `Send.state`.
- The target node's **returned** updates still merge into shared state via the normal reducer mechanism — only input is overridden, not output.
- **Atomic commit**: every branch's update lands in shared state together, in one shot, only once **every** branch succeeds. If any branch raises, the dispatch commits nothing — not even the branches that finished first — and exactly **one** checkpoint is saved for the whole dispatch (never one per branch). This is what makes the dispatch safe to reason about at any N, and safe to resume: a checkpoint can never represent "some but not all branches done."
- One side effect of atomic commit: while a branch is running, `on_node_end`/`on_branch_end` and its `node_result` stream event see shared state as it looked **before** the dispatch (not yet including this branch's own update, or any sibling's) — the update becomes visible from the next node onward. Plain nodes and static fan-out are unaffected.
- **Branch correlation**: since every branch of a dispatch shares the same `node` name, every event/hook for it also carries `branch_index` (0-based), `branch_total`, and `dispatch_id` (shared within one dispatch, unique across dispatches — even to the same node) — via `FlowEngineStreamResponse` for `stream()`, and via the `on_branch_start`/`on_branch_end` hooks (see [Hooks](#hooks)). These are `None` for plain nodes and static fan-out, which already have distinct node names.
- **`max_steps` sizing**: a dispatch counts as (router) + N branches + (reduce node) steps. Sizing `max_steps` for N items needs roughly `N + 2` steps, not just `N`.
- **Known limitation**: one routing call can only fan out to one target node — fanning out to *different* nodes from a single `Send` call isn't supported.

Full runnable example: [`llmfy/example/flowengine_dynamic_fan_out_example.py`](llmfy/example/flowengine_dynamic_fan_out_example.py).

## Retry & timeout

```python
from llmfy import RetryPolicy

flow.add_node(
    "flaky",
    flaky_node,
    retry=RetryPolicy(max_attempts=3, backoff_seconds=0.05, retry_on=(FlakyServiceError,)),
)
flow.add_node("slow", slow_node, timeout=0.05)  # seconds, per attempt
```

`RetryPolicy(max_attempts=1, backoff_seconds=0.0, backoff_multiplier=2.0, retry_on=(Exception,))` — `max_attempts=1` (default) means no retry. Backoff is exponential: `backoff_seconds * backoff_multiplier ** (attempt - 1)`. `retry_on` restricts which exception types are retried; anything else propagates immediately. On exhaustion, a failure raises `NodeExecutionException` (or `NodeTimeoutException` for a timed-out attempt) — never the node's raw exception. `GraphValidationException` (a contract error, e.g. "node has no function") is never retried or wrapped, since it isn't transient.

Under a dynamic fan-out, retry/timeout apply **per branch** independently.

## Hooks

```python
from llmfy import FlowEngineHooks

flow = FlowEngine(
    AppState,
    hooks=FlowEngineHooks(
        on_node_start=lambda name, state: ...,             # (node_name, state)
        on_node_end=lambda name, state, updates: ...,       # (node_name, state, updates)
        on_error=lambda name, exc: ...,                     # (node_name, exception)
        on_branch_start=lambda name, idx, total, state: ...,          # Send branches only
        on_branch_end=lambda name, idx, total, state, updates: ...,   # Send branches only
    ),
)
```

- `on_node_start`/`on_node_end`/`on_error` fire for every node, including each branch of a dynamic fan-out. Any callback may be sync or async.
- `on_branch_start`/`on_branch_end` fire **in addition to** the above, only for dynamic-fan-out branches — use them when you need to tell branches of one dispatch apart (their `node_name` is identical for all of them). `state` in `on_branch_end` is the pre-commit snapshot described in [Dynamic fan-out](#dynamic-fan-out-send).
- Hooks are set once per `FlowEngine` instance (an engine-level observability concern), not per call.

## Tool-calling helpers

`llmfy/flow_engine/helper/` ships the pieces a tool-calling agent node needs, so you don't hand-roll them (used throughout `flowengine_agent_*.py`):

```python
from llmfy import tools_node, tools_stream_node, tool_trim_messages
```

**`tools_node(messages, registry)`** — non-streaming tool executor. Reads `messages[-1].tool_calls`, runs each through a `ToolRegistry` (`llmfy_core/tools/`), and returns the resulting `list[Message]` (`role=Role.TOOL`) to merge back into state:

```python
def tools_executor(state: AppState) -> dict:
    results = tools_node(messages=state["messages"], registry=tool_registry)
    return {"messages": results}

flow.add_node("tools", tools_executor)
```

**`tools_stream_node(messages, registry)`** — the streaming counterpart: a generator yielding one `ToolNodeStreamResponse(type=EXECUTING, name=..., arguments=...)` before each tool call runs, then `ToolNodeStreamResponse(type=RESULT, name=..., arguments=..., result=<Message>)` after — so a `stream=True` node can surface "now calling X" to the UI before the result is known:

```python
async def tools_executor(state: AppState):
    for event in tools_stream_node(messages=state["messages"], registry=tool_registry):
        if event.type == ToolNodeStreamType.EXECUTING:
            yield NodeStreamResponse(type=NodeStreamType.STREAM, content=event)
        elif event.type == ToolNodeStreamType.RESULT:
            yield NodeStreamResponse(
                type=NodeStreamType.RESULT, content=event, state={"messages": [event.result]}
            )

flow.add_node("tools", tools_executor, stream=True)
```

Both helpers execute every pending tool call from the last message **sequentially, in one node**. To run several tool calls from the same turn **concurrently** instead, dispatch one `Send` per call to a single-tool-call node (see [Dynamic fan-out](#dynamic-fan-out-send) and `flowengine_agent_advance_example.py`) rather than looping inside a `tools_node`-style node.

**`tool_trim_messages(messages)`** — a history-trimming function safe to call before an LLM call (or before checkpointing) that always keeps an in-flight tool-call cycle intact: it protects every message from the last tool-calling `ASSISTANT` message onward whenever a tool call is still pending a result, or the last message *is* a tool result, and otherwise aggressively trims down to `messages[-1:]`. This is what closes the "long-running session" gap noted under [Checkpointing & resume](#checkpointing--resume) — run it on `state["messages"]` before it's persisted:

```python
def main_orchestrator(state: AppState) -> dict:
    msgs = tool_trim_messages(state["messages"])
    response = llm.chat(msgs)
    return {"messages": [response.messages[-1]]}
```

Never trims mid tool-call — see the function's docstring in `helper/messages_trimmer/messages_trimmer.py` for the full forward-scan mechanism and worked examples (including parallel tool calls in one `ASSISTANT` message).

## Checkpointing & resume

```python
from llmfy import InMemoryCheckpointer  # or RedisCheckpointer, SQLCheckpointer

checkpointer = InMemoryCheckpointer()          # optional: max_checkpoints_per_session=N
flow = FlowEngine(AppState, checkpointer=checkpointer)
flow.add_node("step", step_node)
flow.add_edge(START, "step")
flow.add_edge("step", END)
flow.build()

session_id = "user-123"
result_1 = await flow.invoke(apply_state={"step": 0}, session_id=session_id)
# Later call, same session_id: continues from the last checkpoint;
# apply_state merges into the resumed state via each field's reducer.
result_2 = await flow.invoke(apply_state={"note": "resumed"}, session_id=session_id)

await flow.get_state(session_id)                 # current checkpointed state, or None
await flow.list_checkpoints(session_id, limit=10)
await flow.get_checkpoint(session_id, checkpoint_id=None)  # latest, or a specific one
await flow.delete_checkpoints(session_id)
await flow.reset_session(session_id)             # deletes all checkpoints for the session
```

**Checkpoint metadata**: every `Checkpoint` returned by `get_checkpoint`/`list_checkpoints`/`get_state` carries a `CheckpointMetadata`:

| Field | Meaning |
|---|---|
| `checkpoint_id` | Unique id for this one checkpoint. |
| `session_id` | The session it belongs to — spans every call (fresh and resumed) across the session's whole lifetime. |
| `run_id` | Identifies the single `invoke()`/`stream()` call that produced this checkpoint. Fresh every call, resumed or not — every checkpoint saved during that one call shares it, so checkpoints can be grouped by run to see which nodes executed together (see `RedisCheckpointer.list_by_run` below). |
| `created_at` | When this checkpoint was saved (UTC). |
| `node` | The node whose completion produced this checkpoint. |
| `prev_node` | Whichever node's completion led to `node` running — `START` (`"__start__"`) if `node` is the first node executed in this *run*. |
| `step` | This checkpoint's position within its run — **always starts at 0 for a new run, even a resumed one**; not a cumulative count across the session. |
| `updated_fields` | The state keys this checkpoint's update touched — the node's own returned keys, or (for a dynamic fan-out's aggregate checkpoint) the union of every branch's keys. A cheap "what changed here" without diffing full state snapshots. |
| `attempt` | The 1-indexed try that succeeded (`1` = no retry needed). `None` for a dynamic fan-out's aggregate checkpoint — several branches, each with its own possibly-different attempt count, are committed together, so no single attempt number applies. |
| `dispatch_id` | Set only on the single checkpoint saved for a dynamic (`Send`) fan-out dispatch (correlates with the same id on that dispatch's `FlowEngineStreamResponse` events); `None` for every other checkpoint. |

**Per-run semantics, resumed or not**: `max_steps` guards one `invoke()`/`stream()` call, not a `session_id`'s cumulative lifetime — a session reused across many calls (e.g. a multi-turn chat loop, one call per turn) never accumulates a shared step count that trips the limit on an otherwise-fine call. A resumed call picks up the graph *position* to continue from (which node runs next), but gets its own fresh `run_id` and step budget — so `prev_node` is `START` for the first node of a resumed run too, even when that node's actual graph predecessor is some other node from a *previous* run (e.g. resuming after a crash mid-graph: the node that resumes is, from its own run's perspective, the first one executed).

Backends:

| Checkpointer | Notes |
|---|---|
| `InMemoryCheckpointer(max_checkpoints_per_session=None, ttl_seconds=None)` | Keeps live Python objects (deep-copied) — no serialization, arbitrary state types work without registration. Lost on process exit. No `compress`/`encryption_key` (nothing ever becomes bytes here to compress or encrypt). |
| `RedisCheckpointer(redis_url=..., prefix="llmfy_checkpoint:", ttl=None, compress=False, encryption_key=None, max_state_bytes=25MiB, max_checkpoints_per_session=None)` | Requires `pip install "llmfy[redis]"`. Crosses a JSON serialization boundary. `ttl` is a **sliding, whole-session** expiry (refreshed via `EXPIRE` on every save) — it does not prune individual old checkpoints out of an active session; use `max_checkpoints_per_session` for that. Also has `await checkpointer.list_by_run(session_id, run_id, limit=100)` — every checkpoint saved during one run, newest first, read from its own Redis index (`session:{session_id}:run:{run_id}`) rather than walking the session's whole checkpoint history, so it stays cheap for a long-lived session with many past runs. Redis-only for now — `list_checkpoints(session_id)` + a client-side filter on `c.metadata.run_id` works on any backend. |
| `SQLCheckpointer(connection_string, echo=False, compress=False, encryption_key=None, max_state_bytes=25MiB, max_checkpoints_per_session=None, ttl_seconds=None)` | Requires `pip install "llmfy[sqlalchemy]"` (or the matching DB driver extra). Supports async and sync SQLAlchemy connection strings (Postgres/MySQL/SQLite). `state` is a binary column (`LargeBinary`/`LONGBLOB`), not text — see below. |

**Compression & encryption at rest** (`SQLCheckpointer`/`RedisCheckpointer` only): both off by default. `compress=True` zlib-compresses state before writing (pure storage/write-size win); `encryption_key=Fernet.generate_key()` Fernet-encrypts it (requires `pip install "llmfy[crypto]"`). Order is compress-then-encrypt on write, reversed on read. With both off, `RedisCheckpointer` stores `state` as plain, human-readable JSON rather than base64 — base64-wrapping only kicks in once there's binary (compressed and/or encrypted) output to carry; `SQLCheckpointer` always uses a binary column regardless, so its `state` is inspectable JSON bytes either way unless compression/encryption is turned on. `max_state_bytes` (default 25 MiB) rejects an oversized checkpoint with `CheckpointPayloadTooLargeException` before it's written, instead of silently exhausting storage — this check runs regardless of `compress`/`encryption_key`. **Caveat**: `SQLCheckpointer(echo=True)` logs bound parameters as-is (i.e. plaintext unless `encryption_key` is set) — never enable it in production.

**Retention** (count-cap and/or TTL, combinable, on all three backends): `max_checkpoints_per_session` keeps only the newest N checkpoints per session, pruned right after each save; `ttl_seconds` (`ttl` on `RedisCheckpointer`, with the sliding-expiry caveat above) drops checkpoints older than a configured age. Both are `None` by default — unbounded, matching pre-existing behavior for anyone not opting in.

**Not yet addressed by any of the above**: retention bounds *how many* checkpoints exist and encryption/compression bound *storage cost per byte*, but neither bounds the *size of a single checkpoint* — since every checkpoint is still a full snapshot of `ctx.state`, a long-running session (e.g. an ever-growing `messages` list, especially with multi-turn tool calling) still produces larger and larger individual checkpoints over time. Bounding that (e.g. a pre-checkpoint trim hook using [`tool_trim_messages`](#tool-calling-helpers), or delta/event-sourcing checkpointing) is a deliberately deferred follow-up, not yet implemented.

**Custom types in state** (Pydantic `BaseModel` or `@dataclass`) crossing a Redis/SQL checkpointer's serialization boundary must be registered explicitly — an unregistered type raises `CheckpointDeserializationException` rather than silently degrading:

```python
flow = FlowEngine(AppState, checkpointer=redis_checkpointer, types=[MyCustomType])
# or:
flow.register_type(MyCustomType)
```

`InMemoryCheckpointer` never needs this (`requires_serialization = False`) since it keeps live objects.

**`session_id` validation**: a caller-supplied `session_id` must match `^[A-Za-z0-9_.:-]{1,255}$` (safe as both a SQL primary key and a Redis key-namespace segment) or `invoke()`/`stream()` raises `InvalidSessionIdException` — checked once, before any checkpointer is touched. An empty string or omitted `session_id` falls back to an auto-generated `uuid4()`, same as before this check existed.

**Resume limitations** (documented, not silent):
- Resuming right after a conditional node re-derives nothing from the checkpoint alone (the condition function isn't re-evaluated) — the caller restarts from `START` instead.
- Resuming mid a *static* fan-out (multiple branches in flight when the process stopped) isn't supported; resume takes the first target of the completed node.
- A *dynamic* (`Send`) fan-out's atomic commit (see above) means this hazard doesn't apply there: a checkpoint only ever exists for a fully-committed dispatch, never a partial one.

## Streaming

```python
async for response in flow.stream(apply_state=None, session_id=None, max_steps=None):
    # response: FlowEngineStreamResponse
    print(response.type, response.node, response.content)
```

`FlowEngineStreamResponse` fields: `type` (`"start" | "stream" | "result" | "error"`), `node`, `content`, `state` (snapshot after this event — see the atomic-commit caveat above for `Send` branches), `error`, and `branch_index` / `branch_total` / `dispatch_id` (all `None` outside a dynamic fan-out).

A **streaming node** (`add_node(..., stream=True)`) must be an (async) generator yielding `NodeStreamResponse(type=NodeStreamType.STREAM, content=...)` chunks, ending with exactly one `NodeStreamResponse(type=NodeStreamType.RESULT, content=..., state={...})` carrying the node's update dict in `state`:

```python
from llmfy import NodeStreamResponse, NodeStreamType

async def generate(state: AppState):
    full = ""
    async for chunk in llm_stream(state["prompt"]):
        full += chunk
        yield NodeStreamResponse(type=NodeStreamType.STREAM, content=chunk)
    yield NodeStreamResponse(type=NodeStreamType.RESULT, content=full, state={"answer": full})

flow.add_node("generate", generate, stream=True)
```

`invoke()` internally drains the same event stream and returns only the final state — use it when you don't need incremental output.

## Visualizing a workflow

```python
flow.build()
print(flow.details())     # plain-text summary of nodes/edges
print(flow.visualize())   # mermaid.ink URL rendering the graph
```

## Exceptions

All FlowEngine errors subclass `LLMfyException`:

| Exception | Raised when |
|---|---|
| `GraphValidationException` | Structural/usage errors: undefined node reference, missing START/END path, reserved name, self-loop, calling `invoke()`/`stream()` before `build()`, a `Send` targeting an undeclared node or mixing target nodes, a checkpoint operation without a checkpointer configured. |
| `NodeExecutionException` | A node raised and all configured retry attempts were exhausted (`node_name`, `attempt`). |
| `NodeTimeoutException` | A node exceeded its `timeout` on its final attempt (subclass of `NodeExecutionException`, adds `timeout_seconds`). |
| `StepLimitExceededException` | A run exceeded `max_steps` without reaching END — usually an unintended loop in a conditional edge (`step`, `max_steps`, `node_name`). |
| `CheckpointDeserializationException` | A checkpoint referenced a custom type not registered via `register_type`/`types=[...]` (`type_name`, `field_name`). |
| `CheckpointPayloadTooLargeException` | A checkpoint's serialized state exceeded the checkpointer's `max_state_bytes` (`session_id`, `size_bytes`, `max_bytes`). |
| `InvalidSessionIdException` | A caller-supplied `session_id` failed the checkpointer identifier allow-list (`session_id`). |

## Further examples

| File | Demonstrates |
|---|---|
| `flowengine_basic_example.py` | Linear flow, conditional loop, reducer. |
| `flowengine_fan_out_example.py` | Static fan-out/fan-in. |
| `flowengine_dynamic_fan_out_example.py` | `Send`, atomic commit, branch correlation via `stream()`. |
| `flowengine_retry_hooks_example.py` | `RetryPolicy`, `timeout`, `FlowEngineHooks`. |
| `flowengine_checkpointer_example.py` | `InMemoryCheckpointer`, resume, `get_state`/`reset_session`. |
| `flowengine_sql_checkpointer_example.py` / `flowengine_redis_checkpointer_example.py` | Persistent checkpointer backends. |
| `flowengine_agent_example.py` / `flowengine_agent_stream_example.py` | An LLM tool-calling agent built on FlowEngine, batch and streaming, via `tools_node`/`tools_stream_node`. |
| `flowengine_agent_advance_example.py` | Same agent, driven with `stream()`, combining dynamic fan-out (`Send`, one branch per tool call the model requested) with static fan-out/join (`audit_log` + `safety_check` after every round) and `tool_trim_messages`. |



## DEBUG manual on redis

For code, prefer `await checkpointer.list_by_run(session_id, run_id)` (see [Checkpointing & resume](#checkpointing--resume)) — it does exactly the ZRANGE-then-GET-each below and returns real `Checkpoint` objects (codec-decoded). The manual form here is for poking around directly in `redis-cli`/Redis Insight, e.g. to inspect raw stored JSON. Both read from the same key, `{prefix}session:{session_id}:run:{run_id}` — keep the prefix consistent between the two calls below (it must match the `prefix` the `RedisCheckpointer` was constructed with).

### by run id

```lua
EVAL "local ids = redis.call('ZRANGE', KEYS[1], 0, -1); local parts = {}; for _, id in ipairs(ids) do local value = redis.call('GET', ARGV[1] .. id); if value then table.insert(parts, value); end; end; return '[' .. table.concat(parts, ',') .. ']'" 1 <your_prefix_flowengine>:session:<your_session_id>:run:<your_run_id> <your_prefix_flowengine>:checkpoint:
```