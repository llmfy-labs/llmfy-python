"""Unit tests for llmfy/flow_engine/node/node.py."""

from llmfy.flow_engine.execution.policy import RetryPolicy
from llmfy.flow_engine.node.node import END, START, Node, NodeType


class TestNodeDefaults:
    def test_minimal_construction(self):
        node = Node(name="n1", node_type=NodeType.FUNCTION)
        assert node.name == "n1"
        assert node.func is None
        assert node.sources == []
        assert node.targets == []
        assert node.stream is False
        assert node.retry is None
        assert node.timeout is None

    def test_sources_and_targets_are_independent_lists_across_instances(self):
        a = Node(name="a", node_type=NodeType.FUNCTION)
        b = Node(name="b", node_type=NodeType.FUNCTION)
        a.targets.append("x")
        assert b.targets == []


class TestNodePolicyFields:
    def test_retry_policy_stored(self):
        policy = RetryPolicy(max_attempts=3)
        node = Node(name="n1", node_type=NodeType.FUNCTION, retry=policy)
        assert node.retry is policy
        assert node.retry.max_attempts == 3  # type: ignore

    def test_timeout_stored(self):
        node = Node(name="n1", node_type=NodeType.FUNCTION, timeout=30.0)
        assert node.timeout == 30.0


class TestStartEndConstants:
    def test_start_and_end_are_distinct_reserved_names(self):
        assert START == "__start__"
        assert END == "__end__"
        assert START != END


class TestNodeType:
    def test_members(self):
        assert NodeType.START.value == "start"
        assert NodeType.END.value == "end"
        assert NodeType.FUNCTION.value == "function"
        assert NodeType.CONDITIONAL.value == "conditional"
