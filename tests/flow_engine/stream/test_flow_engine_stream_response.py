"""Unit tests for llmfy/flow_engine/stream/flow_engine_stream_response.py."""

from enum import StrEnum

from llmfy.flow_engine.stream.flow_engine_stream_response import (
    FlowEngineStreamResponse,
    FlowEngineStreamType,
)
from llmfy.flow_engine.stream.node_stream_response import NodeStreamType
from llmfy.flow_engine.stream.tool_node_stream_response import ToolNodeStreamType


class TestFlowEngineStreamTypeIsStrEnum:
    def test_is_a_str_enum_like_the_other_stream_types(self):
        # All three stream type enums should derive from StrEnum consistently
        # (this used to be inconsistent: FlowEngineStreamType derived from a
        # plain Enum re-exported through node.node).
        assert issubclass(FlowEngineStreamType, StrEnum)
        assert issubclass(NodeStreamType, StrEnum)
        assert issubclass(ToolNodeStreamType, StrEnum)

    def test_values(self):
        assert FlowEngineStreamType.START == "start"
        assert FlowEngineStreamType.STREAM == "stream"
        assert FlowEngineStreamType.RESULT == "result"
        assert FlowEngineStreamType.ERROR == "error"

    def test_compares_equal_to_plain_string(self):
        assert FlowEngineStreamType.RESULT == "result"
        assert "result" == FlowEngineStreamType.RESULT


class TestFlowEngineStreamResponse:
    def test_all_fields_default_to_none(self):
        response = FlowEngineStreamResponse()
        assert response.type is None
        assert response.node is None
        assert response.content is None
        assert response.state is None
        assert response.error is None

    def test_construction_with_values(self):
        response = FlowEngineStreamResponse(
            type=FlowEngineStreamType.RESULT, node="a", content="x", state={"k": 1}
        )
        assert response.type == "result"
        assert response.node == "a"
        assert response.content == "x"
        assert response.state == {"k": 1}
