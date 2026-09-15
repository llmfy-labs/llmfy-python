"""Unit tests for llmfy/llmfy_core/messages/message_buffer_builder.py."""

import pytest

from llmfy.exception.llmfy_exception import LLMfyException
from llmfy.llmfy_core.messages.message import Message
from llmfy.llmfy_core.messages.message_buffer_builder import MessageBufferBuilder
from llmfy.llmfy_core.messages.role import Role
from llmfy.llmfy_core.messages.tool_call import ToolCall
from llmfy.llmfy_core.model_backend import ModelBackend


@pytest.fixture
def temp() -> MessageBufferBuilder:
    return MessageBufferBuilder()


class TestAddSystemMessage:
    def test_inserted_at_front(self, temp: MessageBufferBuilder):
        temp.add_user_message("u1", "hi")
        temp.add_system_message("be nice")
        assert temp.messages[0].role == Role.SYSTEM
        assert temp.messages[0].content == "be nice"
        assert temp.messages[1].id == "u1"

    def test_second_call_pushes_first_system_message_to_index_1(
        self, temp: MessageBufferBuilder
    ):
        temp.add_system_message("first")
        temp.add_system_message("second")
        # No dedup: both remain, most-recent at the very front.
        assert temp.messages[0].content == "second"
        assert temp.messages[1].content == "first"


class TestAddUserMessage:
    def test_appended_with_given_id(self, temp: MessageBufferBuilder):
        temp.add_user_message("u1", "hello")
        assert temp.messages[-1].id == "u1"
        assert temp.messages[-1].role == Role.USER
        assert temp.messages[-1].content == "hello"


class TestAddAssistantMessage:
    def test_appended_message(self, temp: MessageBufferBuilder):
        temp.add_assistant_message("a1", content="hi there")
        assert temp.messages[-1].role == Role.ASSISTANT
        assert temp.messages[-1].content == "hi there"

    def test_mutates_tool_call_request_call_id_in_place(self, temp: MessageBufferBuilder):
        tool_call = ToolCall(
            tool_call_id="call_1", request_call_id="stale", name="fn", arguments={}
        )
        temp.add_assistant_message("a1", tool_calls=[tool_call])
        # add_assistant_message overwrites request_call_id on the caller's
        # own ToolCall object — a documented side effect, not a copy.
        assert tool_call.request_call_id == "a1"
        assert temp.messages[-1].tool_calls[0].request_call_id == "a1"  # type: ignore

    def test_no_tool_calls_does_not_raise(self, temp: MessageBufferBuilder):
        temp.add_assistant_message("a1")
        assert temp.messages[-1].tool_calls is None


class TestAddToolMessage:
    def test_delegates_to_real_formatter_and_appends(self, temp: MessageBufferBuilder):
        temp.add_tool_message(
            id="t1",
            tool_call_id="call_1",
            name="get_weather",
            result="sunny",
            backend=ModelBackend.OPENAI_CHAT,
            request_call_id="req_1",
        )
        assert temp.messages[-1].role == Role.TOOL
        assert temp.messages[-1].tool_results == ["sunny"]

    def test_unsupported_backend_raises_llmfy_exception(
        self, temp: MessageBufferBuilder, monkeypatch
    ):
        monkeypatch.setattr(
            MessageBufferBuilder, "_get_formatter", classmethod(lambda cls, backend: None)
        )
        with pytest.raises(LLMfyException, match="Unsupported model backend"):
            temp.add_tool_message(
                id="t1",
                tool_call_id="call_1",
                name="fn",
                result="x",
                backend=ModelBackend.OPENAI_CHAT,
            )


class TestGetMessages:
    def test_unsupported_backend_raises(self, temp: MessageBufferBuilder, monkeypatch):
        monkeypatch.setattr(
            MessageBufferBuilder, "_get_formatter", classmethod(lambda cls, backend: None)
        )
        with pytest.raises(LLMfyException, match="Unsupported model backend"):
            temp.get_messages(backend=ModelBackend.OPENAI_CHAT)

    def test_formats_every_message_in_order(self, temp: MessageBufferBuilder):
        temp.add_user_message("u1", "hi")
        temp.add_assistant_message("a1", content="hello back")
        formatted = temp.get_messages(backend=ModelBackend.OPENAI_CHAT)
        assert [m["role"] for m in formatted] == ["user", "assistant"]

    def test_format_message_is_cached_per_id(self, temp: MessageBufferBuilder, monkeypatch):
        formatter = MessageBufferBuilder._get_formatter(ModelBackend.OPENAI_CHAT)
        call_count = {"n": 0}
        original = formatter.format_message  # type: ignore

        def counting_format_message(message):
            call_count["n"] += 1
            return original(message)

        monkeypatch.setattr(formatter, "format_message", counting_format_message)

        temp.add_user_message("u1", "hi")
        temp.get_messages(backend=ModelBackend.OPENAI_CHAT)
        temp.get_messages(backend=ModelBackend.OPENAI_CHAT)
        temp.get_messages(backend=ModelBackend.OPENAI_CHAT)

        assert call_count["n"] == 1

    def test_new_message_after_cache_populated_only_formats_the_new_one(
        self, temp: MessageBufferBuilder, monkeypatch
    ):
        formatter = MessageBufferBuilder._get_formatter(ModelBackend.OPENAI_CHAT)
        call_count = {"n": 0}
        original = formatter.format_message  # type: ignore

        def counting_format_message(message):
            call_count["n"] += 1
            return original(message)

        monkeypatch.setattr(formatter, "format_message", counting_format_message)

        temp.add_user_message("u1", "hi")
        temp.get_messages(backend=ModelBackend.OPENAI_CHAT)
        temp.add_user_message("u2", "again")
        temp.get_messages(backend=ModelBackend.OPENAI_CHAT)

        assert call_count["n"] == 2

    def test_cache_evicted_after_clear(self, temp: MessageBufferBuilder):
        temp.add_system_message("sys")
        temp.add_user_message("u1", "hi")
        temp.get_messages(backend=ModelBackend.OPENAI_CHAT)
        assert len(temp._formatted_cache[ModelBackend.OPENAI_CHAT]) == 2

        temp.clear()  # keeps only the system message
        temp.get_messages(backend=ModelBackend.OPENAI_CHAT)
        assert set(temp._formatted_cache[ModelBackend.OPENAI_CHAT].keys()) == {
            msg.id for msg in temp.messages
        }


class TestGetInstanceMessages:
    def test_returns_live_reference_not_a_copy(self, temp: MessageBufferBuilder):
        temp.add_user_message("u1", "hi")
        messages = temp.get_instance_messages()
        messages.append(Message(role=Role.USER, content="mutated externally"))
        assert len(temp.messages) == 2


class TestClear:
    def test_no_system_message_leaves_history_empty(self, temp: MessageBufferBuilder):
        temp.add_user_message("u1", "hi")
        temp.clear()
        assert temp.messages == []

    def test_keeps_the_frontmost_system_message(self, temp: MessageBufferBuilder):
        temp.add_system_message("first")
        temp.add_user_message("u1", "hi")
        temp.add_system_message("second")  # now at index 0
        temp.clear()
        assert len(temp.messages) == 1
        assert temp.messages[0].content == "second"
