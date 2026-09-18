"""Unit tests for llmfy/flow_engine/graph/validation.py."""

import pytest

from llmfy.exception.llmfy_exception import GraphValidationException
from llmfy.flow_engine.edge.edge import Edge
from llmfy.flow_engine.graph.graph_builder import build_graph
from llmfy.flow_engine.graph.validation import validate_workflow
from llmfy.flow_engine.node.node import END, START, Node, NodeType


def make_nodes(*names: str) -> dict[str, Node]:
    nodes = {START: Node(name=START, node_type=NodeType.START), END: Node(name=END, node_type=NodeType.END)}
    for name in names:
        nodes[name] = Node(name=name, node_type=NodeType.FUNCTION)
    return nodes


def validate(nodes, edges):
    graph = build_graph(nodes, edges)
    validate_workflow(nodes, edges, graph)


class TestStartAndEndPaths:
    def test_valid_linear_workflow_passes(self):
        nodes = make_nodes("a")
        edges = [Edge(START, "a"), Edge("a", END)]
        validate(nodes, edges)  # should not raise

    def test_missing_start_edge_raises(self):
        nodes = make_nodes("a")
        edges = [Edge("a", END)]
        with pytest.raises(GraphValidationException, match="No edge from START"):
            validate(nodes, edges)

    def test_missing_end_edge_raises(self):
        nodes = make_nodes("a", "b")
        edges = [Edge(START, "a"), Edge("a", "b")]
        with pytest.raises(GraphValidationException, match="No edge to END"):
            validate(nodes, edges)

    def test_end_reachable_via_conditional_edge_passes(self):
        nodes = make_nodes("a")

        def cond(_state):
            return END

        edges = [Edge(START, "a"), Edge("a", [END], condition=cond)]
        validate(nodes, edges)


class TestUndefinedNodes:
    def test_undefined_target_raises(self):
        nodes = make_nodes("a")
        edges = [Edge(START, "a"), Edge("a", "ghost"), Edge("ghost", END)]
        with pytest.raises(GraphValidationException, match="not defined"):
            validate(nodes, edges)

    def test_undefined_source_raises(self):
        nodes = make_nodes("a")
        edges = [Edge(START, "a"), Edge("a", END), Edge("ghost", END)]
        with pytest.raises(GraphValidationException, match="not defined"):
            validate(nodes, edges)


class TestMixedOutgoingEdgeKinds:
    def test_regular_and_conditional_from_same_source_raises(self):
        nodes = make_nodes("a", "b", "c")

        def cond(_state):
            return "c"

        edges = [
            Edge(START, "a"),
            Edge("a", "b"),
            Edge("a", ["b", "c"], condition=cond),
            Edge("b", END),
            Edge("c", END),
        ]
        with pytest.raises(GraphValidationException, match="both a conditional edge and a regular edge"):
            validate(nodes, edges)


class TestJoinValidation:
    def test_join_with_all_regular_predecessors_passes(self):
        nodes = make_nodes("fetch", "a", "b", "combine")
        edges = [
            Edge(START, "fetch"),
            Edge("fetch", ["a", "b"]),
            Edge("a", "combine"),
            Edge("b", "combine"),
            Edge("combine", END),
        ]
        validate(nodes, edges)

    def test_join_with_a_conditional_predecessor_raises(self):
        nodes = make_nodes("fetch", "a", "b", "c", "combine")

        def cond(_state):
            return "combine"

        edges = [
            Edge(START, "fetch"),
            Edge("fetch", ["a", "b"]),
            Edge("a", "combine"),
            Edge("b", "combine"),
            Edge("c", ["combine", END], condition=cond),
            Edge("combine", END),
        ]
        with pytest.raises(GraphValidationException, match="deadlock the join barrier"):
            validate(nodes, edges)


class TestUnreachableNodeWarning:
    def test_unreachable_node_warns_but_does_not_raise(self):
        nodes = make_nodes("a", "orphan")
        edges = [Edge(START, "a"), Edge("a", END)]
        with pytest.warns(UserWarning, match="orphan"):
            validate(nodes, edges)
