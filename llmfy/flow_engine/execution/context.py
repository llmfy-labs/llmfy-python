import asyncio
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class ExecutionContext:
    """All per-call mutable state for one `invoke()`/`stream()` run.

    Created fresh by `FlowEngine.invoke()`/`stream()` for every call and
    never stored on the `FlowEngine` instance itself — this is what lets
    two concurrent calls on the same `FlowEngine` run safely: each owns a
    private `state` dict and step counter, so neither can race the other.

    `run_id` identifies this one call, fresh whether it starts at START or
    resumes from a checkpoint — every `Checkpoint` saved during this run
    carries it, so checkpoints can be grouped by run to see which nodes
    executed together. `step` always starts at 0 for a new run (even a
    resumed one): `max_steps` guards a single `invoke()`/`stream()` call
    against a runaway loop, not the cumulative lifetime of a `session_id`.

    `lock` serializes the small critical sections that mutate this context
    from concurrent fan-out branches (applying reducer updates, join
    arrival bookkeeping, incrementing `step`) — cheap insurance against
    lost updates, not a bottleneck, since each critical section is a plain
    dict mutation with no `await` inside it.
    """

    session_id: str
    state: dict[str, Any]
    run_id: str = field(default_factory=lambda: str(uuid4()))
    max_steps: int = 100
    step: int = 0
    join_arrivals: dict[str, set[str]] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
