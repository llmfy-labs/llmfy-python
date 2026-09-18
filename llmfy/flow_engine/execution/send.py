from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Send:
    """Dynamically dispatch one parallel execution of `node`, seeded with
    `state` as that invocation's complete input — replacing, not merging
    with, the shared workflow state the routing function saw.

    Return one `Send`, or a `list[Send]`, from a conditional edge's routing
    function (registered via `FlowEngine.add_conditional_edges`) to fan out
    to a runtime-determined number of parallel branches of the SAME target
    node — e.g. one branch per item in a list whose length is only known
    when the routing function runs. `node` must be one of that conditional
    edge's declared `targets`. Every `Send` returned from one routing call
    must target the same `node` — fanning out to different nodes from one
    routing call is not supported.

    The target node's own RETURNED updates still merge into the shared
    workflow state via the normal reducer mechanism — only the node's
    INPUT is replaced by `state`, not its output.

    Atomic commit: a dispatch's branches all merge into shared state
    together, in one shot, only once EVERY branch succeeds — if any branch
    raises, the others' updates are discarded too and `ctx.state` ends up
    completely untouched by the dispatch (see
    `flowengine_dynamic_fan_out_example.py`). One consequence: while a
    branch is running, `FlowEngineHooks.on_node_end`/`on_branch_end` and
    its `node_result` stream event see shared state as it looked BEFORE
    this dispatch (not yet including this branch's own update, or any
    sibling's) — the update becomes visible only from the next node
    onward, once the dispatch has committed. Checkpointing follows the
    same all-or-nothing rule: exactly one checkpoint is saved for the
    whole dispatch (after it commits), never one per branch — so a
    checkpoint can never represent "some but not all branches done,"
    which is what made resuming mid-dispatch unsafe in an earlier version.

    Branch correlation: every event/hook for one branch carries
    `branch_index` (0-based), `branch_total`, and `dispatch_id` (shared by
    every branch of one dispatch, distinct across separate dispatches even
    to the same node) — via `FlowEngineStreamResponse` for `stream()`
    consumers, and via `FlowEngineHooks.on_branch_start`/`on_branch_end`
    for hook consumers. Plain nodes and static fan-out (`add_edge` with
    multiple targets) leave these `None`; their branches already have
    distinct `node` names, so they don't need it.

    Known v1 limitation: one routing call may only fan out to ONE target
    node (see the class docstring above) — not fixed by the above.
    """

    node: str
    state: dict[str, Any]
