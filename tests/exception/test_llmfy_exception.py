"""Unit tests for the LLMfyException hierarchy (llmfy/exception/llmfy_exception.py)."""

import pytest

from llmfy.exception.llmfy_exception import (
    AuthenticationException,
    CheckpointDeserializationException,
    CheckpointPayloadTooLargeException,
    ContentFilterException,
    GraphValidationException,
    InvalidRequestException,
    InvalidSessionIdException,
    LLMfyException,
    ModelErrorException,
    ModelNotFoundException,
    NodeExecutionException,
    NodeTimeoutException,
    PermissionDeniedException,
    QuotaExceededException,
    RateLimitException,
    ServiceUnavailableException,
    StepLimitExceededException,
    TimeoutException,
    TimeoutType,
)

# Every leaf subclass except TimeoutException/NodeExecutionException/
# NodeTimeoutException/StepLimitExceededException/
# CheckpointDeserializationException/CheckpointPayloadTooLargeException/
# InvalidSessionIdException has no added behavior — its own
# __init__/__repr__ is inherited unmodified from LLMfyException.
PLAIN_SUBCLASSES = [
    RateLimitException,
    QuotaExceededException,
    InvalidRequestException,
    AuthenticationException,
    PermissionDeniedException,
    ModelNotFoundException,
    ServiceUnavailableException,
    ContentFilterException,
    ModelErrorException,
    GraphValidationException,
]


class TestLLMfyExceptionBase:
    def test_message_only_construction_does_not_raise(self):
        exc = LLMfyException("boom")
        assert exc.message == "boom"
        assert exc.status_code is None
        assert exc.raw_error is None
        assert exc.provider is None

    def test_str_returns_message(self):
        # super().__init__(message) means str(exc) mirrors plain Exception behavior
        exc = LLMfyException("something went wrong")
        assert str(exc) == "something went wrong"

    def test_all_fields_stored(self):
        raw = {"foo": "bar"}
        exc = LLMfyException(
            "boom", status_code=500, raw_error=raw, provider="openai"
        )
        assert exc.message == "boom"
        assert exc.status_code == 500
        assert exc.raw_error is raw
        assert exc.provider == "openai"

    def test_repr_format(self):
        exc = LLMfyException("boom", status_code=429, provider="openai")
        assert repr(exc) == "LLMfyException(message='boom', status_code=429, provider='openai')"

    def test_repr_omits_raw_error(self):
        exc = LLMfyException("boom", raw_error={"secret": "value"})
        assert "secret" not in repr(exc)

    def test_is_a_plain_exception(self):
        assert isinstance(LLMfyException("x"), Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(LLMfyException) as exc_info:
            raise LLMfyException("failure", status_code=400)
        assert exc_info.value.status_code == 400


class TestPlainSubclasses:
    @pytest.mark.parametrize("exc_class", PLAIN_SUBCLASSES)
    def test_subclass_of_llmfy_exception(self, exc_class):
        exc = exc_class("boom")
        assert isinstance(exc, LLMfyException)
        assert isinstance(exc, Exception)

    @pytest.mark.parametrize("exc_class", PLAIN_SUBCLASSES)
    def test_repr_uses_concrete_class_name(self, exc_class):
        exc = exc_class("boom", status_code=1, provider="openai")
        assert repr(exc).startswith(f"{exc_class.__name__}(")

    @pytest.mark.parametrize("exc_class", PLAIN_SUBCLASSES)
    def test_full_field_construction(self, exc_class):
        exc = exc_class("boom", status_code=503, raw_error={"a": 1}, provider="bedrock")
        assert exc.message == "boom"
        assert exc.status_code == 503
        assert exc.raw_error == {"a": 1}
        assert exc.provider == "bedrock"


class TestTimeoutException:
    def test_default_timeout_type_is_none(self):
        exc = TimeoutException("timed out")
        assert exc.timeout_type is None

    @pytest.mark.parametrize(
        "timeout_type",
        [TimeoutType.CONNECT, TimeoutType.READ, TimeoutType.WRITE, TimeoutType.POOL, TimeoutType.MODEL],
    )
    def test_stores_timeout_type(self, timeout_type):
        exc = TimeoutException("timed out", timeout_type=timeout_type)
        assert exc.timeout_type is timeout_type

    def test_inherits_base_fields(self):
        exc = TimeoutException(
            "timed out", status_code=408, provider="openai", timeout_type=TimeoutType.READ
        )
        assert exc.message == "timed out"
        assert exc.status_code == 408
        assert exc.provider == "openai"

    def test_repr_does_not_include_timeout_type(self):
        # TimeoutException does not override __repr__, so it stays the base
        # LLMfyException format — timeout_type is silently absent from repr().
        exc = TimeoutException("timed out", timeout_type=TimeoutType.CONNECT)
        assert "timeout_type" not in repr(exc)
        assert "connect" not in repr(exc)

    def test_is_llmfy_exception(self):
        assert isinstance(TimeoutException("x"), LLMfyException)


class TestTimeoutTypeEnum:
    def test_values(self):
        assert TimeoutType.CONNECT.value == "connect"
        assert TimeoutType.READ.value == "read"
        assert TimeoutType.WRITE.value == "write"
        assert TimeoutType.POOL.value == "pool"
        assert TimeoutType.MODEL.value == "model"


class TestNodeExecutionException:
    def test_stores_node_name_and_attempt(self):
        exc = NodeExecutionException("boom", node_name="call_llm", attempt=3)
        assert exc.node_name == "call_llm"
        assert exc.attempt == 3
        assert exc.message == "boom"

    def test_inherits_base_fields(self):
        exc = NodeExecutionException(
            "boom", node_name="n", attempt=1, status_code=500, provider="openai"
        )
        assert exc.status_code == 500
        assert exc.provider == "openai"

    def test_is_llmfy_exception(self):
        assert isinstance(NodeExecutionException("x", node_name="n", attempt=1), LLMfyException)


class TestNodeTimeoutException:
    def test_stores_timeout_seconds_and_base_fields(self):
        exc = NodeTimeoutException(
            "timed out", node_name="call_llm", attempt=2, timeout_seconds=30.0
        )
        assert exc.node_name == "call_llm"
        assert exc.attempt == 2
        assert exc.timeout_seconds == 30.0

    def test_is_node_execution_exception(self):
        exc = NodeTimeoutException("x", node_name="n", attempt=1, timeout_seconds=5.0)
        assert isinstance(exc, NodeExecutionException)
        assert isinstance(exc, LLMfyException)


class TestStepLimitExceededException:
    def test_stores_fields(self):
        exc = StepLimitExceededException(
            "too many steps", session_id="s1", step=101, max_steps=100, node_name="loop"
        )
        assert exc.session_id == "s1"
        assert exc.step == 101
        assert exc.max_steps == 100
        assert exc.node_name == "loop"

    def test_node_name_defaults_to_none(self):
        exc = StepLimitExceededException("x", session_id="s1", step=1, max_steps=1)
        assert exc.node_name is None

    def test_is_llmfy_exception(self):
        assert isinstance(
            StepLimitExceededException("x", session_id="s1", step=1, max_steps=1),
            LLMfyException,
        )


class TestCheckpointDeserializationException:
    def test_stores_type_name_and_field_name(self):
        exc = CheckpointDeserializationException(
            "not registered", type_name="pkg.mod.Foo", field_name="messages"
        )
        assert exc.type_name == "pkg.mod.Foo"
        assert exc.field_name == "messages"

    def test_field_name_defaults_to_none(self):
        exc = CheckpointDeserializationException("x", type_name="pkg.mod.Foo")
        assert exc.field_name is None

    def test_is_llmfy_exception(self):
        assert isinstance(
            CheckpointDeserializationException("x", type_name="t"), LLMfyException
        )


class TestCheckpointPayloadTooLargeException:
    def test_stores_fields(self):
        exc = CheckpointPayloadTooLargeException(
            "too big", session_id="s1", size_bytes=200, max_bytes=100
        )
        assert exc.session_id == "s1"
        assert exc.size_bytes == 200
        assert exc.max_bytes == 100

    def test_is_llmfy_exception(self):
        assert isinstance(
            CheckpointPayloadTooLargeException(
                "x", session_id="s1", size_bytes=2, max_bytes=1
            ),
            LLMfyException,
        )


class TestInvalidSessionIdException:
    def test_stores_session_id(self):
        exc = InvalidSessionIdException("bad id", session_id="../etc/passwd")
        assert exc.session_id == "../etc/passwd"

    def test_is_llmfy_exception(self):
        assert isinstance(
            InvalidSessionIdException("x", session_id="s1"), LLMfyException
        )
