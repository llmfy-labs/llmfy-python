"""Unit tests for llmfy/llmfy_core/retry/llm_retry_policy.py."""

import pytest

from llmfy.exception.llmfy_exception import (
    AuthenticationException,
    ContentFilterException,
    InvalidRequestException,
    LLMfyException,
    ModelErrorException,
    QuotaExceededException,
    RateLimitException,
    ServiceUnavailableException,
    TimeoutException,
    TimeoutType,
)
from llmfy.llmfy_core.messages.tool_call import ToolCall
from llmfy.llmfy_core.responses.ai_response import AIResponse
from llmfy.llmfy_core.retry import (
    LLMRetryPolicy,
    is_capacity_error,
    is_empty_response,
    is_retryable_llm_error,
)
from llmfy.llmfy_core.retry.llm_retry_policy import backoff_delay


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ModelErrorException("model failed", 424), True),
        (TimeoutException("model timeout", 408, timeout_type=TimeoutType.MODEL), True),
        # Network timeouts were already retried by the SDK.
        (TimeoutException("read timeout", timeout_type=TimeoutType.READ), False),
        (TimeoutException("connect timeout", timeout_type=TimeoutType.CONNECT), False),
        (TimeoutException("unknown timeout"), False),
        # Capacity errors are governed by throttle_wait, not max_attempts.
        (RateLimitException("throttled", 429), False),
        (ServiceUnavailableException("unavailable", 503), False),
        (InvalidRequestException("bad request", 400), False),
        (AuthenticationException("denied", 403), False),
        (ContentFilterException("blocked"), False),
        (ValueError("bug"), False),
    ],
)
def test_is_retryable_llm_error(error: Exception, expected: bool):
    assert is_retryable_llm_error(error) is expected


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RateLimitException("throttled", 429), True),
        # Bedrock ModelNotReadyException is mapped to ServiceUnavailable/429.
        (ServiceUnavailableException("not ready", 429), True),
        (ServiceUnavailableException("internal", 500), True),
        (ServiceUnavailableException("unavailable", 503), True),
        # Anthropic OverloadedError has no dedicated mapping: plain LLMfyException/529.
        (LLMfyException("overloaded", 529), True),
        (LLMfyException("unknown 503", 503), True),
        (QuotaExceededException("quota", 402), False),
        (ModelErrorException("model failed", 424), False),
        (InvalidRequestException("bad request", 400), False),
        (LLMfyException("no status"), False),
        (ValueError("bug"), False),
    ],
)
def test_is_capacity_error(error: Exception, expected: bool):
    assert is_capacity_error(error) is expected


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (AIResponse(), True),
        (AIResponse(content=""), True),
        (AIResponse(content="  \n"), True),
        (AIResponse(content="", thinking="thought only"), True),
        (AIResponse(content="hi"), False),
        (
            AIResponse(
                tool_calls=[
                    ToolCall(tool_call_id="t1", request_call_id="r1", name="f", arguments={})
                ]
            ),
            False,
        ),
    ],
)
def test_is_empty_response(response: AIResponse, expected: bool):
    assert is_empty_response(response) is expected


class TestPolicyValidation:
    def test_defaults(self):
        policy = LLMRetryPolicy()
        assert policy.max_attempts == 3
        assert policy.throttle_wait == 0
        assert policy.retry_on_empty is True
        assert policy.on_empty_exhausted == "return"
        assert policy.retry_stream is True

    def test_max_attempts_must_be_positive(self):
        with pytest.raises(ValueError, match="max_attempts"):
            LLMRetryPolicy(max_attempts=0)

    @pytest.mark.parametrize(
        "field",
        ["base_delay", "max_delay", "throttle_wait", "throttle_base_delay", "throttle_max_delay"],
    )
    def test_delays_must_not_be_negative(self, field: str):
        with pytest.raises(ValueError, match=field):
            LLMRetryPolicy(**{field: -1})

    def test_on_empty_exhausted_is_checked(self):
        with pytest.raises(ValueError, match="on_empty_exhausted"):
            LLMRetryPolicy(on_empty_exhausted="ignore")  # type: ignore[arg-type]

    def test_policy_is_immutable(self):
        policy = LLMRetryPolicy()
        with pytest.raises(AttributeError):
            policy.max_attempts = 5  # type: ignore[misc]


class TestBackoff:
    @pytest.mark.parametrize(("retry", "nominal"), [(1, 1.0), (2, 2.0), (3, 4.0), (4, 8.0), (5, 8.0)])
    def test_doubles_up_to_the_cap_with_jitter(self, retry: int, nominal: float):
        for _ in range(50):
            delay = backoff_delay(retry, base_delay=1.0, max_delay=8.0)
            assert nominal * 0.5 <= delay <= nominal

    def test_zero_base_delay_never_sleeps(self):
        assert backoff_delay(3, base_delay=0.0, max_delay=8.0) == 0.0
