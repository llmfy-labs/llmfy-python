"""Unit tests for llmfy/flow_engine/helper/tools_node/tools_node.py.

`tools_node`/`tools_stream_node` are thin integration helpers between
FlowEngine and `ToolRegistry` — their external contract (constructing
`Message(role=Role.TOOL, ...)` results satisfying `Message`'s own
`tool_results`-required-for-TOOL-role invariant) must keep working
unchanged under the rewritten async execution model.
"""

from llmfy.flow_engine.helper.tools_node.tools_node import tools_node, tools_stream_node
from llmfy.flow_engine.stream.tool_node_stream_response import ToolNodeStreamType
from llmfy.llmfy_core.messages.message import Message
from llmfy.llmfy_core.messages.role import Role
from llmfy.llmfy_core.tools.tool import Tool
from llmfy.llmfy_core.tools.tool_registry import ToolRegistry
from tests.conftest import FakeAIModel


@Tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def make_registry() -> ToolRegistry:
    return ToolRegistry(funcs=[add], model=FakeAIModel())


def assistant_with_tool_call(tool_call_id: str = "tc1") -> Message:
    from llmfy.llmfy_core.messages.tool_call import ToolCall

    return Message(
        role=Role.ASSISTANT,
        content=None,
        tool_calls=[
            ToolCall(
                tool_call_id=tool_call_id,
                request_call_id="r1",
                name="add",
                arguments={"a": 1, "b": 2},
            )
        ],
    )


class TestToolsNode:
    async def test_executes_pending_tool_call_and_returns_tool_message(self):
        messages = [assistant_with_tool_call()]
        registry = make_registry()

        results = tools_node(messages, registry)

        assert len(results) == 1
        result = results[0]
        assert result.role == Role.TOOL
        assert result.tool_call_id == "tc1"
        assert result.tool_results == ["3"]

    async def test_no_tool_calls_returns_empty_list(self):
        messages = [Message(role=Role.ASSISTANT, content="hi")]
        registry = make_registry()

        assert tools_node(messages, registry) == []

    async def test_does_not_mutate_input_messages(self):
        messages = [assistant_with_tool_call()]
        registry = make_registry()

        tools_node(messages, registry)

        assert messages[0].tool_calls[0].tool_call_id == "tc1"  # type: ignore


class TestToolsStreamNode:
    async def test_yields_executing_then_result(self):
        messages = [assistant_with_tool_call()]
        registry = make_registry()

        events = list(tools_stream_node(messages, registry))

        assert [e.type for e in events] == [
            ToolNodeStreamType.EXECUTING,
            ToolNodeStreamType.RESULT,
        ]
        assert events[0].name == "add"
        assert events[1].result.role == Role.TOOL  # type: ignore
        assert events[1].result.tool_results == ["3"]  # type: ignore

    async def test_no_tool_calls_yields_nothing(self):
        messages = [Message(role=Role.ASSISTANT, content="hi")]
        registry = make_registry()

        assert list(tools_stream_node(messages, registry)) == []
