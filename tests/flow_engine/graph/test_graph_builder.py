"""Unit tests for llmfy/flow_engine/graph/graph_builder.py."""

from llmfy.flow_engine.edge.edge import Edge
from llmfy.flow_engine.graph.graph_builder import build_graph
from llmfy.flow_engine.node.node import END, START, Node, NodeType


def make_nodes(*names: str) -> dict[str, Node]:
    nodes = {START: Node(name=START, node_type=NodeType.START), END: Node(name=END, node_type=NodeType.END)}
    for name in names:
        nodes[name] = Node(name=name, node_type=NodeType.FUNCTION)
    return nodes


class TestLinearGraph:
    def test_targets_of_single_edge(self):
        nodes = make_nodes("a", "b")
        edges = [Edge(START, "a"), Edge("a", "b"), Edge("b", END)]
        graph = build_graph(nodes, edges)

        assert graph.targets_of("a") == ["b"]
        assert graph.targets_of("b") == [END]

    def test_no_predecessors_for_single_chain(self):
        nodes = make_nodes("a", "b")
        edges = [Edge(START, "a"), Edge("a", "b"), Edge("b", END)]
        graph = build_graph(nodes, edges)

        assert graph.predecessors["b"] == {"a"}
        assert not graph.is_join("b")


class TestFanOut:
    def test_list_target_produces_multiple_targets(self):
        nodes = make_nodes("fetch", "a", "b")
        edges = [Edge(START, "fetch"), Edge("fetch", ["a", "b"])]
        graph = build_graph(nodes, edges)

        assert graph.targets_of("fetch") == ["a", "b"]

    def test_two_separate_add_edge_calls_merge_into_one_fan_out(self):
        nodes = make_nodes("fetch", "a", "b")
        edges = [Edge(START, "fetch"), Edge("fetch", "a"), Edge("fetch", "b")]
        graph = build_graph(nodes, edges)

        assert graph.targets_of("fetch") == ["a", "b"]

    def test_duplicate_target_is_deduplicated(self):
        nodes = make_nodes("fetch", "a")
        edges = [Edge(START, "fetch"), Edge("fetch", "a"), Edge("fetch", "a")]
        graph = build_graph(nodes, edges)

        assert graph.targets_of("fetch") == ["a"]


class TestFanIn:
    def test_join_node_has_multiple_predecessors(self):
        nodes = make_nodes("fetch", "a", "b", "combine")
        edges = [
            Edge(START, "fetch"),
            Edge("fetch", ["a", "b"]),
            Edge("a", "combine"),
            Edge("b", "combine"),
            Edge("combine", END),
        ]
        graph = build_graph(nodes, edges)

        assert graph.predecessors["combine"] == {"a", "b"}
        assert graph.is_join("combine")

    def test_single_predecessor_is_not_a_join(self):
        nodes = make_nodes("a", "b")
        edges = [Edge(START, "a"), Edge("a", "b"), Edge("b", END)]
        graph = build_graph(nodes, edges)

        assert not graph.is_join("b")


class TestStartEdge:
    def test_start_is_not_counted_as_a_predecessor(self):
        nodes = make_nodes("a", "b")
        edges = [Edge(START, "a"), Edge("a", "b"), Edge("b", END)]
        graph = build_graph(nodes, edges)

        assert graph.predecessors["a"] == set()

    def test_entry_node_looped_back_to_is_not_a_join(self):
        """Regression: an agent-loop node (e.g. `main`) is reached once from
        START and repeatedly from a loop-back edge (e.g. `tools -> main`).
        Counting START as a real predecessor alongside the loop-back source
        made `is_join` see 2 predecessors, but only the loop-back source
        ever "arrives" through the join-tracking path (the initial START
        call bypasses it) — so the node would wait forever for a second
        arrival that never comes, silently stopping the workflow."""
        nodes = make_nodes("main", "tools")
        edges = [
            Edge(START, "main"),
            Edge("tools", "main"),
            Edge("main", ["tools", END], condition=lambda _state: "tools"),
        ]
        graph = build_graph(nodes, edges)

        assert graph.predecessors["main"] == {"tools"}
        assert not graph.is_join("main")


class TestConditionalEdges:
    def test_conditional_targets_are_not_counted_as_predecessors(self):
        nodes = make_nodes("a", "b", "c")

        def cond(_state):
            return "b"

        edges = [
            Edge(START, "a"),
            Edge("a", ["b", "c"], condition=cond),
            Edge("b", END),
            Edge("c", END),
        ]
        graph = build_graph(nodes, edges)

        assert graph.predecessors.get("b", set()) == set()
        assert graph.predecessors.get("c", set()) == set()
        assert not graph.is_join("b")

    def test_targets_of_conditional_node_is_empty_static_list(self):
        nodes = make_nodes("a", "b", "c")

        def cond(_state):
            return "b"

        edges = [Edge(START, "a"), Edge("a", ["b", "c"], condition=cond)]
        graph = build_graph(nodes, edges)

        assert graph.targets_of("a") == []
        assert graph.is_conditional("a")

    def test_is_conditional_false_for_regular_edge(self):
        nodes = make_nodes("a", "b")
        edges = [Edge(START, "a"), Edge("a", "b")]
        graph = build_graph(nodes, edges)

        assert not graph.is_conditional("a")
