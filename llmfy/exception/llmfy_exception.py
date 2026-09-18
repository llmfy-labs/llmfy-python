"""
LLMfy Exception Handling

Documentation References:

- AWS Bedrock Converse API: [https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
- OpenAI Python SDK: [https://github.com/openai/openai-python#handling-errors](https://github.com/openai/openai-python#handling-errors)
- Google Gen AI SDK: [https://github.com/googleapis/python-genai#error-handling](https://github.com/googleapis/python-genai#error-handling)

Example of catching and inspecting an error:
```python
try:
    # _call_llmfy
    pass
except LLMfyException as e:
    print(f"Error: {e.message}")
    print(f"Status Code: {e.status_code}")
    print(f"Provider: {e.provider}")
    print(f"Raw Error: {e.raw_error}")

    # Check specific exception type
    if isinstance(e, RateLimitException):
        print("Rate limited! Implement backoff...")
    elif isinstance(e, TimeoutException):
        print("Request timed out! Retry...")
```
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Deferred: `llmfy_exception.py` loads very early (`llmfy/__init__.py`
    # imports `.exception` before `.llmfy_core`), so a real top-level import
    # here would trigger `llmfy_core/__init__.py` — which pulls in every
    # provider model, several of which import back `from llmfy import
    # LLMfyException` — before `llmfy.LLMfyException` itself exists yet.
    from llmfy.llmfy_core.service_provider import ServiceProvider


class TimeoutType(Enum):
    CONNECT = "connect"  # could not establish a connection to the endpoint
    READ = "read"        # connected but timed out waiting for response bytes
    WRITE = "write"      # timed out sending the request
    POOL = "pool"        # timed out waiting for a connection slot from the pool
    MODEL = "model"      # model-side processing timeout (Bedrock ModelTimeoutException)


class LLMfyException(Exception):
    """
    Base LLMfy Exception

    Example of catching and inspecting an error:
    ```python
    try:
        # _call_llmfy
        pass
    except LLMfyException as e:
        print(f"Error: {e.message}")
        print(f"Status Code: {e.status_code}")
        print(f"Provider: {e.provider}")
        print(f"Raw Error: {e.raw_error}")

        # Check specific exception type
        if isinstance(e, RateLimitException):
            print("Rate limited! Implement backoff...")
        elif isinstance(e, TimeoutException):
            print("Request timed out! Retry...")
    ```
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        raw_error: Any | None = None,
        provider: ServiceProvider | str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.raw_error = raw_error
        self.provider = provider

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"status_code={self.status_code}, "
            f"provider={self.provider!r})"
        )


class RateLimitException(LLMfyException):
    """Rate limit exceeded"""

    pass


class QuotaExceededException(LLMfyException):
    """Quota/usage limit exceeded"""

    pass


class TimeoutException(LLMfyException):
    """Request timed out"""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        raw_error: Any | None = None,
        provider: ServiceProvider | str | None = None,
        timeout_type: TimeoutType | None = None,
    ):
        super().__init__(message, status_code, raw_error, provider)
        self.timeout_type = timeout_type


class InvalidRequestException(LLMfyException):
    """Invalid request parameters"""

    pass


class AuthenticationException(LLMfyException):
    """Authentication failed"""

    pass


class PermissionDeniedException(LLMfyException):
    """Permission denied"""

    pass


class ModelNotFoundException(LLMfyException):
    """Model not found or unavailable"""

    pass


class ServiceUnavailableException(LLMfyException):
    """Service temporarily unavailable"""

    pass


class ContentFilterException(LLMfyException):
    """Content blocked by safety filters"""

    pass


class ModelErrorException(LLMfyException):
    """Model processing error"""

    pass


class GraphValidationException(LLMfyException):
    """FlowEngine workflow graph is structurally invalid or misused.

    Covers build-time and usage-order errors: undefined node references,
    missing START/END path, reserved node names, self-loops, a join node
    with a conditional predecessor, calling invoke()/stream() before
    build(), or a checkpoint operation without a checkpointer configured.
    """

    pass


class NodeExecutionException(LLMfyException):
    """A FlowEngine node raised and all configured retry attempts were exhausted."""

    def __init__(
        self,
        message: str,
        node_name: str,
        attempt: int,
        status_code: int | None = None,
        raw_error: Any | None = None,
        provider: ServiceProvider | str | None = None,
    ):
        super().__init__(message, status_code, raw_error, provider)
        self.node_name = node_name
        self.attempt = attempt


class NodeTimeoutException(NodeExecutionException):
    """A FlowEngine node exceeded its configured timeout on its final attempt."""

    def __init__(
        self,
        message: str,
        node_name: str,
        attempt: int,
        timeout_seconds: float,
        status_code: int | None = None,
        raw_error: Any | None = None,
        provider: ServiceProvider | str | None = None,
    ):
        super().__init__(message, node_name, attempt, status_code, raw_error, provider)
        self.timeout_seconds = timeout_seconds


class StepLimitExceededException(LLMfyException):
    """A FlowEngine run exceeded its configured max_steps without reaching END."""

    def __init__(
        self,
        message: str,
        session_id: str,
        step: int,
        max_steps: int,
        node_name: str | None = None,
        status_code: int | None = None,
        raw_error: Any | None = None,
        provider: ServiceProvider | str | None = None,
    ):
        super().__init__(message, status_code, raw_error, provider)
        self.session_id = session_id
        self.step = step
        self.max_steps = max_steps
        self.node_name = node_name


class CheckpointDeserializationException(LLMfyException):
    """FlowEngine checkpoint state referenced a type that is not registered.

    Raised on both save (an unregistered custom type in the state) and load
    (a stored checkpoint tags a type the current process hasn't registered
    via `FlowEngine(types=[...])` / `flow.register_type(...)`) — the engine
    fails closed rather than silently degrading to a raw dict.
    """

    def __init__(
        self,
        message: str,
        type_name: str,
        field_name: str | None = None,
        status_code: int | None = None,
        raw_error: Any | None = None,
        provider: ServiceProvider | str | None = None,
    ):
        super().__init__(message, status_code, raw_error, provider)
        self.type_name = type_name
        self.field_name = field_name


class CheckpointPayloadTooLargeException(LLMfyException):
    """A checkpoint's serialized state exceeded the checkpointer's configured
    `max_state_bytes` and was rejected before being written to storage."""

    def __init__(
        self,
        message: str,
        session_id: str,
        size_bytes: int,
        max_bytes: int,
        status_code: int | None = None,
        raw_error: Any | None = None,
        provider: ServiceProvider | str | None = None,
    ):
        super().__init__(message, status_code, raw_error, provider)
        self.session_id = session_id
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes


class InvalidSessionIdException(LLMfyException):
    """A `session_id` passed to `FlowEngine.invoke()`/`stream()` failed the
    checkpointer identifier allow-list (safe for use as a SQL primary key and
    a Redis key-namespace segment)."""

    def __init__(
        self,
        message: str,
        session_id: str,
        status_code: int | None = None,
        raw_error: Any | None = None,
        provider: ServiceProvider | str | None = None,
    ):
        super().__init__(message, status_code, raw_error, provider)
        self.session_id = session_id
