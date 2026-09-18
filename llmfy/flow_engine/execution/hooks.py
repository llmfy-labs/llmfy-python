from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class FlowEngineHooks:
    """Observability hooks for a `FlowEngine` instance.

    Each callback may be sync or async (dispatched the same way node and
    condition functions already are). Set once at
    `FlowEngine(..., hooks=FlowEngineHooks(...))` construction — hooks are
    an engine-level observability concern, not a per-call option.

    Deliberately minimal: exactly these three lifecycle points (node start,
    node end, error) cover tracing/logging for real LLM-agent workflows
    without growing into a generic event-bus API.
    """

    on_node_start: Callable[[str, dict[str, Any]], Any] | None = None
    """Called with (node_name, state) right before a node executes."""

    on_node_end: Callable[[str, dict[str, Any], dict[str, Any]], Any] | None = None
    """Called with (node_name, state, updates) right after a node succeeds."""

    on_error: Callable[[str, Exception], Any] | None = None
    """Called with (node_name, exception) when a node's execution ultimately fails
    (after retries are exhausted). The exception still propagates afterward."""

    on_branch_start: Callable[[str, int, int, dict[str, Any]], Any] | None = None
    """Called with (node_name, branch_index, branch_total, state) right before
    one dynamic-fan-out (`Send`) branch executes — fired in ADDITION to
    `on_node_start`, which also fires for the same branch with just
    (node_name, state). Use this when you need to tell branches of one
    dispatch apart (e.g. "3 of 200"); every branch of one dispatch shares
    the same `node_name`, so `on_node_start` alone can't distinguish them.
    Not fired for plain nodes or static fan-out (`add_edge` with multiple
    targets) — those already have distinct `node_name`s per branch."""

    on_branch_end: Callable[[str, int, int, dict[str, Any], dict[str, Any]], Any] | None = None
    """Called with (node_name, branch_index, branch_total, state, updates)
    right after one dynamic-fan-out (`Send`) branch succeeds — fired in
    ADDITION to `on_node_end`. `state` is shared state as of just BEFORE
    this dispatch's updates are committed: a dynamic-fan-out dispatch
    commits atomically, only once every branch succeeds, so no branch
    (including this one) can see any branch's updates reflected in `state`
    yet — see `execution/send.py` for why. `updates` is this branch's own
    (not-yet-committed) return value."""
