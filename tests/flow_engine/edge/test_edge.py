"""Unit tests for llmfy/flow_engine/edge/edge.py."""

from llmfy.flow_engine.edge.edge import Edge


class TestTargetsNormalization:
    def test_string_target_becomes_single_item_list(self):
        edge = Edge(source="a", targets="b")
        assert edge.targets == ["b"]

    def test_list_target_is_kept_as_is(self):
        edge = Edge(source="a", targets=["b", "c"])
        assert edge.targets == ["b", "c"]

    def test_single_item_list_stays_a_list(self):
        edge = Edge(source="a", targets=["b"])
        assert edge.targets == ["b"]


class TestCondition:
    def test_condition_defaults_to_none(self):
        edge = Edge(source="a", targets="b")
        assert edge.condition is None

    def test_condition_is_stored(self):
        def cond(state):
            return "b"

        edge = Edge(source="a", targets=["b", "c"], condition=cond)
        assert edge.condition is cond


class TestTargetMap:
    def test_defaults_to_none(self):
        edge = Edge(source="a", targets=["b", "c"], condition=lambda s: "b")
        assert edge.target_map is None

    def test_stored_independently_of_targets(self):
        edge = Edge(
            source="a",
            targets=["b", "c"],
            condition=lambda s: "Pass",
            target_map={"Pass": "b", "Fail": "c"},
        )
        assert edge.target_map == {"Pass": "b", "Fail": "c"}
        assert edge.targets == ["b", "c"]
