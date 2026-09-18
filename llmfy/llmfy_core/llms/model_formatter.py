from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Deferred: only used for type hints here. A real top-level import would
    # make this module's own import order matter (whichever provider package
    # happens to import `ModelFormatter` first would trigger the `messages`
    # package from inside `model_formatter`'s own partial import, causing a
    # circular-import error) — fragile, and not something import-sorting
    # tools know to preserve. Deferring it removes the runtime dependency
    # entirely, so no ordering constraint exists at all.
    from llmfy.llmfy_core.messages.message import Message


class ModelFormatter(ABC):
    """ModelFormatter.

    Register all derivated intances from this class to:
        - `MessageBufferBuilder` class at `llmfy/llmfy_core/messages/message_buffer_builder.py`
        - `Tool` class at `llmfy/chat/tools/tool.py`

    Args:
        ABC (_type_): _description_
    """

    @abstractmethod
    def format_message(self, message: Message) -> dict:
        pass

    @abstractmethod
    def format_tool_function(
        self, func_metadata: dict, type_mapping: dict[Any, str]
    ) -> dict:
        pass

    @abstractmethod
    def format_tool_message(
        self,
        messages: list[Message],
        id: str,
        tool_call_id: str,
        name: str,
        result: str,
        request_call_id: str | None = None,
    ) -> list[Message]:
        pass
