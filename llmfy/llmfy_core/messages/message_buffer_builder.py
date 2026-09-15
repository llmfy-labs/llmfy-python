from typing import Any

from llmfy.exception.llmfy_exception import LLMfyException
from llmfy.llmfy_core.llms.model_formatter import ModelFormatter
from llmfy.llmfy_core.messages.content import Content
from llmfy.llmfy_core.messages.message import Message
from llmfy.llmfy_core.messages.role import Role
from llmfy.llmfy_core.messages.tool_call import ToolCall
from llmfy.llmfy_core.model_backend import ModelBackend


class MessageBufferBuilder:
    """MessageBufferBuilder class. History only per request, not saved to memory."""

    # Populated lazily by `_get_formatter` on first use, not at import time:
    # eagerly importing the provider formatters here would pull in each
    # provider's package `__init__.py` (and its concrete Model class), which
    # imports back from `llms.base_ai_model` and creates a circular import.
    _formatters: dict[ModelBackend, ModelFormatter] | None = None

    def __init__(self):
        self.messages: list[Message] = []
        # Formatted-message cache: backend -> {Message.id -> formatted dict}.
        #
        # Safe to cache "format once, reuse forever" even though
        # `format_tool_message` (Anthropic/Bedrock) can mutate an existing
        # Message's `tool_results` in place when merging parallel tool
        # results: that merge always targets the *current* assistant turn's
        # `request_call_id`, which is a fresh uuid per turn (see
        # `add_assistant_message`) and is never reused for a past message.
        # So by the time a message is read here (i.e. `get_messages` is
        # called), every message already in `self.messages` is settled and
        # will never be mutated again — only genuinely new messages need
        # formatting.
        self._formatted_cache: dict[ModelBackend, dict[str, dict[str, Any]]] = {}

    @classmethod
    def _get_formatter(cls, backend: ModelBackend) -> ModelFormatter | None:
        if cls._formatters is None:
            from llmfy.llmfy_core.llms.anthropic.messages.anthropic_messages_formatter import (
                AnthropicMessagesFormatter,
            )
            from llmfy.llmfy_core.llms.bedrock.converse.bedrock_converse_formatter import (
                BedrockConverseFormatter,
            )
            from llmfy.llmfy_core.llms.google.generate.googleai_generate_formatter import (
                GoogleAIGenerateFormatter,
            )
            from llmfy.llmfy_core.llms.openai.chat.openai_chat_formatter import (
                OpenAIChatFormatter,
            )
            from llmfy.llmfy_core.llms.openai.responses.openai_responses_formatter import (
                OpenAIResponsesFormatter,
            )

            cls._formatters = {
                ModelBackend.OPENAI_CHAT: OpenAIChatFormatter(),
                ModelBackend.OPENAI_RESPONSES: OpenAIResponsesFormatter(),
                ModelBackend.BEDROCK_CONVERSE: BedrockConverseFormatter(),
                ModelBackend.GOOGLE_GENERATE: GoogleAIGenerateFormatter(),
                ModelBackend.ANTHROPIC_MESSAGES: AnthropicMessagesFormatter(),
            }
        return cls._formatters.get(backend)

    def add_system_message(self, content: str) -> None:
        self.messages.insert(0, Message(role=Role.SYSTEM, content=content))

    def add_user_message(
        self, id: str, content: str | None | list[Content] | None
    ) -> None:
        self.messages.append(Message(id=id, role=Role.USER, content=content))

    def add_assistant_message(
        self,
        id: str,
        content: str | None | list[Content] | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        # Update request call id by parent
        if tool_calls:
            for tool_call in tool_calls:
                tool_call.request_call_id = id

        self.messages.append(
            Message(id=id, role=Role.ASSISTANT, content=content, tool_calls=tool_calls)
        )

    def add_tool_message(
        self,
        id: str,
        tool_call_id: str,
        name: str,
        result: str,
        backend: ModelBackend,
        request_call_id: str | None = None,
    ) -> None:
        formatter = self._get_formatter(backend)
        if not formatter:
            raise LLMfyException(f"Unsupported model backend: {backend}")

        formatter.format_tool_message(
            messages=self.messages,
            id=id,
            tool_call_id=tool_call_id,
            name=name,
            request_call_id=request_call_id,
            result=result,
        )

    def get_messages(self, backend: ModelBackend) -> list[dict[str, Any]]:
        formatter = self._get_formatter(backend)
        if not formatter:
            raise LLMfyException(f"Unsupported model backend: {backend}")

        cache = self._formatted_cache.setdefault(backend, {})

        # Drop entries for messages no longer in history (e.g. after `clear()`
        # dropped everything but the system message) so the cache can't hold
        # stale entries. `LLMfy` itself never calls `clear()` anymore — each
        # `invoke`/`chat` call builds its own fresh `MessageBufferBuilder`
        # instead of reusing and clearing a shared one (see `llmfy.py`) — but
        # `clear()` stays a public method any other caller may still use.
        current_ids = {msg.id for msg in self.messages}
        for stale_id in cache.keys() - current_ids:
            del cache[stale_id]

        formatted = []
        for msg in self.messages:
            if msg.id not in cache:
                cache[msg.id] = formatter.format_message(msg)
            formatted.append(cache[msg.id])
        return formatted

    def get_instance_messages(self) -> list[Message]:
        # return [msg for msg in self.messages if msg.role != Role.SYSTEM]
        return self.messages

    def clear(self) -> None:
        system_message = next(
            (msg for msg in self.messages if msg.role == Role.SYSTEM), None
        )
        self.messages.clear()
        if system_message:
            self.messages.append(system_message)
