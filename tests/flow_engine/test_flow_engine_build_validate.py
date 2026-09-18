"""Unit tests for llmfy/flow_engine/flow_engine.py — construction, state
schema/reducer extraction, add_node/add_edge/add_conditional_edges
validation, and build()."""

from typing import Annotated, TypedDict

import pytest

from llmfy.exception.llmfy_exception import GraphValidationException
from llmfy.flow_engine.edge.edge import Edge
from llmfy.flow_engine.flow_engine import FlowEngine
from llmfy.flow_engine.node.node import END, START, NodeType


def add_reducer(old, new):
    return (old or 0) + new


class SimpleState(TypedDict):
    count: int


class ReducerState(TypedDict):
    total: Annotated[int, add_reducer]
    label: str


async def noop(state):
    return {}


class TestStateSchemaExtraction:
    def test_plain_field_has_no_reducer(self):
        flow = FlowEngine(SimpleState)
        assert flow._reducers["count"] is None
        assert flow._type_hints["count"] is int

    def test_annotated_field_extracts_reducer_and_type(self):
        flow = FlowEngine(ReducerState)
        assert flow._reducers["total"] is add_reducer
        assert flow._type_hints["total"] is int
        assert flow._reducers["label"] is None

    def test_reducer_must_take_exactly_two_params(self):
        def bad_reducer(a, b, c):
            return a

        class BadState(TypedDict):
            x: Annotated[int, bad_reducer]

        with pytest.raises(GraphValidationException, match="exactly 2 parameters"):
            FlowEngine(BadState)

    def test_non_callable_reducer_raises(self):
        class BadState(TypedDict):
            x: Annotated[int, "not callable"]

        with pytest.raises(GraphValidationException, match="must be callable"):
            FlowEngine(BadState)


class TestAddNode:
    def test_reserved_names_rejected(self):
        flow = FlowEngine(SimpleState)
        with pytest.raises(GraphValidationException, match="reserved name"):
            flow.add_node(START, noop)
        with pytest.raises(GraphValidationException, match="reserved name"):
            flow.add_node(END, noop)

    def test_start_and_end_nodes_exist_by_default(self):
        flow = FlowEngine(SimpleState)
        assert START in flow.nodes
        assert END in flow.nodes
        assert flow.nodes[START].node_type == NodeType.START
        assert flow.nodes[END].node_type == NodeType.END


class TestAddEdge:
    def test_start_cannot_be_a_target(self):
        flow = FlowEngine(SimpleState)
        flow.add_node("a", noop)
        with pytest.raises(GraphValidationException, match="START cannot be a target"):
            flow.add_edge("a", START)

    def test_end_cannot_be_a_source(self):
        flow = FlowEngine(SimpleState)
        with pytest.raises(GraphValidationException, match="END cannot be a source"):
            flow.add_edge(END, "a")

    def test_self_loop_rejected(self):
        flow = FlowEngine(SimpleState)
        flow.add_node("a", noop)
        with pytest.raises(GraphValidationException, match="cannot target itself"):
            flow.add_edge("a", "a")

    def test_self_loop_in_fan_out_list_rejected(self):
        flow = FlowEngine(SimpleState)
        flow.add_node("a", noop)
        flow.add_node("b", noop)
        with pytest.raises(GraphValidationException, match="cannot target itself"):
            flow.add_edge("a", ["b", "a"])

    def test_fan_out_list_target_accepted(self):
        flow = FlowEngine(SimpleState)
        flow.add_node("a", noop)
        flow.add_node("b", noop)
        flow.add_node("c", noop)
        flow.add_edge("a", ["b", "c"])

        edge = flow.edges[-1]
        assert isinstance(edge, Edge)
        assert edge.targets == ["b", "c"]

    def test_updates_node_sources_and_targets_bookkeeping(self):
        flow = FlowEngine(SimpleState)
        flow.add_node("a", noop)
        flow.add_node("b", noop)
        flow.add_edge("a", "b")

        assert "b" in flow.nodes["a"].targets
        assert "a" in flow.nodes["b"].sources


class TestAddConditionalEdge:
    def test_start_cannot_be_in_targets(self):
        flow = FlowEngine(SimpleState)
        flow.add_node("a", noop)
        with pytest.raises(GraphValidationException, match="START cannot be a target"):
            flow.add_conditional_edges("a", lambda s: END, [START, END])

    def test_end_cannot_be_source(self):
        flow = FlowEngine(SimpleState)
        with pytest.raises(GraphValidationException, match="END cannot be a source"):
            flow.add_conditional_edges(END, lambda s: "a", ["a"])

    def test_marks_source_node_as_conditional(self):
        flow = FlowEngine(SimpleState)
        flow.add_node("a", noop)
        flow.add_node("b", noop)
        flow.add_conditional_edges("a", lambda s: "b", ["b", END])

        assert flow.nodes["a"].node_type == NodeType.CONDITIONAL

    def test_dict_targets_start_value_rejected(self):
        flow = FlowEngine(SimpleState)
        flow.add_node("a", noop)
        with pytest.raises(GraphValidationException, match="START cannot be a target"):
            flow.add_conditional_edges("a", lambda s: "go", {"go": START})

    def test_dict_targets_builds_edge_with_target_map(self):
        flow = FlowEngine(SimpleState)
        flow.add_node("a", noop)
        flow.add_node("b", noop)
        flow.add_conditional_edges("a", lambda s: "b", {"take_b": "b", "finish": END})

        edge = flow.edges[-1]
        assert edge.target_map == {"take_b": "b", "finish": END}
        assert set(edge.targets) == {"b", END}
        assert flow.nodes["a"].node_type == NodeType.CONDITIONAL


class TestBuild:
    def test_build_returns_self_and_sets_is_built(self):
        flow = FlowEngine(SimpleState)
        flow.add_node("a", noop)
        flow.add_edge(START, "a")
        flow.add_edge("a", END)

        result = flow.build()

        assert result is flow
        assert flow.is_built is True

    def test_invoke_before_build_raises(self):
        import asyncio

        flow = FlowEngine(SimpleState)
        with pytest.raises(GraphValidationException, match="Build first"):
            asyncio.run(flow.invoke())

    def test_missing_start_edge_raises_at_build(self):
        flow = FlowEngine(SimpleState)
        flow.add_node("a", noop)
        flow.add_edge("a", END)
        with pytest.raises(GraphValidationException, match="No edge from START"):
            flow.build()

    def test_missing_end_path_raises_at_build(self):
        flow = FlowEngine(SimpleState)
        flow.add_node("a", noop)
        flow.add_edge(START, "a")
        with pytest.raises(GraphValidationException, match="No edge to END"):
            flow.build()

    def test_undefined_node_reference_raises_at_build(self):
        flow = FlowEngine(SimpleState)
        flow.add_node("a", noop)
        flow.add_edge(START, "a")
        flow.add_edge("a", "ghost")
        flow.add_edge("ghost", END)
        with pytest.raises(GraphValidationException, match="not defined"):
            flow.build()


class TestDetailsAndVisualize:
    def test_details_before_build_raises(self):
        flow = FlowEngine(SimpleState)
        with pytest.raises(GraphValidationException, match="Build first"):
            flow.details()

    def test_details_lists_nodes_and_edges(self):
        flow = FlowEngine(SimpleState)
        flow.add_node("a", noop)
        flow.add_edge(START, "a")
        flow.add_edge("a", END)
        flow.build()

        text = flow.details()
        assert "START -> a" in text
        assert "a" in text

    def test_visualize_returns_mermaid_ink_url(self):
        flow = FlowEngine(SimpleState)
        flow.add_node("a", noop)
        flow.add_edge(START, "a")
        flow.add_edge("a", END)
        flow.build()

        url = flow.visualize()
        assert url.startswith("https://mermaid.ink/img/")
