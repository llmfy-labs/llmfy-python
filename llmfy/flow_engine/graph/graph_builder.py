"""Compiles a workflow's `Node`/`Edge` data into an adjacency structure
built once (in `FlowEngine.build()`), replacing the pre-existing pattern of
linear-scanning `edges` on every single node transition.

Also where multiple `add_edge()` calls from the same source are merged into
one fan-out target list — authoring a fan-out as `add_edge(src, [a, b])` or
as two separate `add_edge(src, a)` / `add_edge(src, b)` calls compiles to
the exact same graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llmfy.flow_engine.edge.edge import Edge
from llmfy.flow_engine.node.node import END, START, Node


@dataclass
class CompiledGraph:
    """Precomputed adjacency/predecessor view over a workflow's nodes and edges."""

    # node_name -> the single Edge describing that node's outgoing connections
    # (either one regular edge, whose targets may fan out to several nodes,
    # or one conditional edge).
    outgoing: dict[str, Edge] = field(default_factory=dict)

    # node_name -> set of predecessor node names reachable via a regular
    # (non-conditional) edge. Conditional-edge targets are intentionally
    # excluded — only one conditional branch executes per pass, so treating
    # every possible conditional target as a "predecessor" would make a
    # join barrier wait forever on a branch that never arrives. `START` is
    # excluded too: the engine's entry call runs the start node directly
    # (`arrived_from=None`), bypassing join-arrival tracking entirely, so
    # `START` never "checks in" — counting it here would make any node that
    # is both the graph's entry point and a loop-back target (e.g. an agent
    # node reached from `START` and looped back to from a tool node) wait
    # forever for a second arrival that can never come.
    predecessors: dict[str, set[str]] = field(default_factory=dict)

    def targets_of(self, node_name: str) -> list[str]:
        """Static outgoing targets of a node with a regular (non-conditional)
        edge. Returns `[]` if the node has no outgoing edge or its outgoing
        edge is conditional (conditional targets are resolved at runtime)."""
        edge = self.outgoing.get(node_name)
        if edge is None or edge.condition is not None:
            return []
        return edge.targets

    def is_conditional(self, node_name: str) -> bool:
        edge = self.outgoing.get(node_name)
        return edge is not None and edge.condition is not None

    def is_join(self, node_name: str) -> bool:
        """A node with more than one non-conditional predecessor — execution
        must wait for every predecessor branch to arrive before running it."""
        return len(self.predecessors.get(node_name, ())) > 1


def build_graph(nodes: dict[str, Node], edges: list[Edge]) -> CompiledGraph:
    """Compile `nodes`/`edges` into a `CompiledGraph`."""
    predecessors: dict[str, set[str]] = {name: set() for name in nodes}
    regular_targets: dict[str, list[str]] = {}
    conditional_edges: dict[str, Edge] = {}

    for edge in edges:
        if edge.condition is not None:
            conditional_edges[edge.source] = edge
            continue

        bucket = regular_targets.setdefault(edge.source, [])
        for target in edge.targets:
            if target not in bucket:
                bucket.append(target)

    outgoing: dict[str, Edge] = {}
    for source, targets in regular_targets.items():
        outgoing[source] = Edge(source=source, targets=targets)
        if source == START:
            continue
        for target in targets:
            if target == END:
                continue
            predecessors.setdefault(target, set()).add(source)

    for source, edge in conditional_edges.items():
        outgoing[source] = edge

    return CompiledGraph(outgoing=outgoing, predecessors=predecessors)
