from collections.abc import Callable
from dataclasses import dataclass


@dataclass(init=False)
class Edge:
    """Represents an edge in the workflow graph.

    The constructor accepts a single node name or a list for `targets`
    (`Edge("a", "b")` / `Edge("a", ["b", "c"])`), but the stored/declared
    field is always the normalized `list[str]` — `__init__` is hand-written
    (dataclass's own generation disabled via `init=False`) so the flexible
    constructor input type and the narrower, always-a-list attribute type
    can differ without every reader of `edge.targets` needing to re-narrow
    a `str | list[str]` union.

    `target_map` is set only for a conditional edge built from a dict
    (`{condition_return_value: target_node}`) — `targets`
    still holds the flat list of possible destination node names (the
    dict's values) for graph structure/validation, while `target_map` is
    consulted at runtime to resolve the condition function's return value
    to the actual next node.
    """
    source: str
    targets: list[str]
    condition: Callable | None = None
    target_map: dict[str, str] | None = None

    def __init__(
        self,
        source: str,
        targets: str | list[str],
        condition: Callable | None = None,
        target_map: dict[str, str] | None = None,
    ) -> None:
        self.source = source
        self.targets = [targets] if isinstance(targets, str) else targets
        self.condition = condition
        self.target_map = target_map
