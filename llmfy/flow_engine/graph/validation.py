"""Build-time structural validation for a `FlowEngine` workflow.

Ported from the pre-rewrite `FlowEngine._validate_workflow`, with two
changes: the old "a node may have at most one outgoing edge unless
conditional" rule is removed (fan-out is now a supported, first-class
pattern — see `graph_builder.py`), and a new join-validation rule is added
to keep fan-in safe.
"""

from __future__ import annotations

import warnings

from llmfy.exception.llmfy_exception import GraphValidationException
from llmfy.flow_engine.edge.edge import Edge
from llmfy.flow_engine.graph.graph_builder import CompiledGraph
from llmfy.flow_engine.node.node import END, START, Node


def validate_workflow(nodes: dict[str, Node], edges: list[Edge], graph: CompiledGraph) -> None:
    """Validate workflow structure before execution.

    Raises:
        GraphValidationException: if the workflow has structural issues.
    """
    _validate_start_has_outgoing_edge(edges)
    _validate_end_is_reachable(edges)

    defined_nodes = set(nodes.keys()) - {START, END}
    all_referenced_nodes = _referenced_nodes(edges)
    _validate_referenced_nodes_are_defined(all_referenced_nodes, defined_nodes)

    _validate_no_mixed_outgoing_edge_kinds(edges)
    _validate_joins(edges, graph)
    _warn_on_unreachable_nodes(defined_nodes, all_referenced_nodes)


def _validate_start_has_outgoing_edge(edges: list[Edge]) -> None:
    if not any(e.source == START for e in edges):
        raise GraphValidationException(
            "No edge from START node. Use flow.add_edge(START, 'node_name')"
        )


def _validate_end_is_reachable(edges: list[Edge]) -> None:
    if not any(END in e.targets for e in edges):
        raise GraphValidationException(
            "No edge to END node. At least one execution path must reach END. "
            "Use flow.add_edge('node_name', END) or include END in conditional targets."
        )


def _referenced_nodes(edges: list[Edge]) -> set[str]:
    referenced: set[str] = set()
    for edge in edges:
        if edge.source not in (START, END):
            referenced.add(edge.source)
        for target in edge.targets:
            if target not in (START, END):
                referenced.add(target)
    return referenced


def _validate_referenced_nodes_are_defined(
    referenced: set[str], defined: set[str]
) -> None:
    undefined = referenced - defined
    if undefined:
        raise GraphValidationException(
            f"Referenced nodes are not defined: {', '.join(sorted(undefined))}. "
            f"Use flow.add_node() to define them."
        )


def _validate_no_mixed_outgoing_edge_kinds(edges: list[Edge]) -> None:
    has_conditional = {e.source for e in edges if e.condition is not None}
    has_regular = {e.source for e in edges if e.condition is None}
    mixed = sorted(has_conditional & has_regular)
    if mixed:
        raise GraphValidationException(
            f"Node(s) {mixed} have both a conditional edge and a regular edge "
            "as source — only one outgoing edge kind is allowed per node. Use "
            "either add_edge() (optionally fanning out to multiple targets) "
            "or add_conditional_edges(), not both, from the same source."
        )


def _validate_joins(edges: list[Edge], graph: CompiledGraph) -> None:
    conditional_targets: dict[str, set[str]] = {}
    for edge in edges:
        if edge.condition is None:
            continue
        for target in edge.targets:
            if target == END:
                continue
            conditional_targets.setdefault(target, set()).add(edge.source)

    for node_name, preds in graph.predecessors.items():
        if len(preds) <= 1:
            continue
        offending = conditional_targets.get(node_name)
        if offending:
            raise GraphValidationException(
                f"Node '{node_name}' is a join (has {len(preds)} predecessors: "
                f"{sorted(preds)}) but also receives a conditional edge from "
                f"{sorted(offending)}. A join's predecessors must all be "
                "non-conditional edges — a conditional edge might route "
                "elsewhere, which would deadlock the join barrier waiting for "
                "a branch that never arrives."
            )


def _warn_on_unreachable_nodes(defined: set[str], referenced: set[str]) -> None:
    unreachable = defined - referenced
    if unreachable:
        warnings.warn(
            f"Some nodes are defined but not reachable: {', '.join(sorted(unreachable))}",
            UserWarning,
            stacklevel=2,
        )
