"""Unit tests for llmfy/flow_engine/helper/messages_trimmer/messages_trimmer.py."""

from llmfy.flow_engine.helper.messages_trimmer.messages_trimmer import (
    tool_trim_messages,
)
from llmfy.llmfy_core.messages.message import Message
from llmfy.llmfy_core.messages.role import Role
from llmfy.llmfy_core.messages.tool_call import ToolCall


def user(text: str) -> Message:
    return Message(role=Role.USER, content=text)


def assistant(text: str = "ok", tool_calls=None) -> Message:
    return Message(role=Role.ASSISTANT, content=text, tool_calls=tool_calls)


def tool_result(
    tool_call_id: str, request_call_id: str = "r1", result: str = "result"
) -> Message:
    return Message(
        role=Role.TOOL,
        tool_call_id=tool_call_id,
        request_call_id=request_call_id,
        tool_results=[result],
    )


def make_tool_call(tool_call_id: str) -> ToolCall:
    return ToolCall(
        tool_call_id=tool_call_id,
        request_call_id="r1",
        name="search",
        arguments={"q": "x"},
    )


class TestToolTrimMessagesSingleMessage:
    def test_single_message_returned_as_is(self):
        messages = [user("hi")]
        assert tool_trim_messages(messages) == messages


class TestToolTrimMessagesNoActiveToolCycle:
    def test_no_tool_calls_keeps_only_last_message(self):
        messages = [user("hello"), assistant("hi"), user("bye"), assistant("later")]
        result = tool_trim_messages(messages)
        assert result == messages[-1:]

    def test_resolved_tool_cycle_followed_by_plain_turns_keeps_only_last_message(self):
        messages = [
            user("search for X"),
            assistant("looking...", tool_calls=[make_tool_call("tc1")]),
            tool_result("tc1"),
            assistant("here is the answer"),
            user("thanks"),
        ]
        result = tool_trim_messages(messages)
        assert result == messages[-1:]


class TestToolTrimMessagesPendingToolCall:
    def test_preserves_pending_tool_call_context(self):
        messages = [
            user("search for X"),
            assistant("looking...", tool_calls=[make_tool_call("tc1")]),
        ]
        result = tool_trim_messages(messages)
        # The assistant message with the pending tool call must survive trimming.
        assert any(m.role == Role.ASSISTANT and m.tool_calls for m in result)
        assert result[-1].tool_calls

    def test_pending_tool_call_anchors_to_last_non_tool_message(self):
        messages = [
            user("first question"),
            assistant("answer 1"),
            user("search for X"),
            assistant("looking...", tool_calls=[make_tool_call("tc1")]),
        ]
        result = tool_trim_messages(messages)
        assert result == messages[2:]

    def test_pending_tool_call_with_nothing_before_it_returns_protected_only(self):
        messages = [assistant("looking...", tool_calls=[make_tool_call("tc1")])]
        # len == 1 short-circuits before the tool-cycle logic
        assert tool_trim_messages(messages) == messages


class TestToolTrimMessagesLastMessageIsToolResult:
    def test_preserves_context_when_last_message_is_tool_result(self):
        messages = [
            user("search for X"),
            assistant("looking...", tool_calls=[make_tool_call("tc1")]),
            tool_result("tc1"),
        ]
        result = tool_trim_messages(messages)
        assert result[-1].role == Role.TOOL
        # Only one message precedes the protected tool cycle and it isn't a
        # TOOL message, so the anchor scan keeps everything.
        assert result == messages

    def test_anchor_skips_past_prior_tool_results_to_last_non_tool_message(self):
        # Two sequential tool cycles: trimming the first cycle must not leave
        # an orphaned TOOL message at the start of the result.
        messages = [
            user("search for X"),
            assistant("looking...", tool_calls=[make_tool_call("tc1")]),
            tool_result("tc1"),
            assistant("based on results...", tool_calls=[make_tool_call("tc2")]),
            tool_result("tc2"),
        ]
        result = tool_trim_messages(messages)
        assert result[0].role != Role.TOOL
        assert result[-1].role == Role.TOOL
        assert result == messages[1:]

    def test_parallel_tool_calls_are_all_preserved(self):
        messages = [
            user("search for X"),
            assistant(
                "looking...",
                tool_calls=[make_tool_call("tc1"), make_tool_call("tc2")],
            ),
            tool_result("tc1"),
            tool_result("tc2"),
        ]
        result = tool_trim_messages(messages)
        assert result == messages
