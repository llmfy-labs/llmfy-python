"""Unit tests for llmfy/flow_engine/execution/send.py (Send)."""

import dataclasses

import pytest

from llmfy.flow_engine.execution.send import Send


class TestSendShape:
    def test_stores_node_and_state(self):
        send = Send(node="process_item", state={"item": "x"})
        assert send.node == "process_item"
        assert send.state == {"item": "x"}

    def test_equality(self):
        assert Send("a", {"x": 1}) == Send("a", {"x": 1})

    def test_inequality_on_node(self):
        assert Send("a", {"x": 1}) != Send("b", {"x": 1})

    def test_inequality_on_state(self):
        assert Send("a", {"x": 1}) != Send("a", {"x": 2})


class TestSendIsFrozen:
    def test_cannot_reassign_node(self):
        send = Send("a", {"x": 1})
        with pytest.raises(dataclasses.FrozenInstanceError):
            send.node = "b"  # type: ignore

    def test_cannot_reassign_state(self):
        send = Send("a", {"x": 1})
        with pytest.raises(dataclasses.FrozenInstanceError):
            send.state = {"y": 2}  # type: ignore
