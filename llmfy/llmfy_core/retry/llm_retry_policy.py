"""Retry policy for the LLM round trips an `LLMfy` instance makes.

Retries happen in two layers:

- The provider SDK retries transport failures within seconds: throttling,
  5xx, dropped connections and read timeouts. Tune that layer per model with
  `SDKRetryConfig`.
- `LLMRetryPolicy` (passed as `LLMfy(..., retry=...)`) retries what the SDK
  treats as final: model-side errors (Bedrock `ModelErrorException`, Bedrock
  `ModelTimeoutException`) and replies with neither text nor tool calls. With a
  `throttle_wait` it also keeps retrying capacity errors (throttling, 503) that
  outlast the SDK's own retries, for up to that many seconds.

The policy wraps each single model round trip, never a whole
`invoke_with_tools`/`chat_with_tools` loop, so a tool that already ran is
never run again.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from llmfy.exception.llmfy_exception import (
    LLMfyException,
    ModelErrorException,
    RateLimitException,
    ServiceUnavailableException,
    TimeoutException,
    TimeoutType,
)
from llmfy.llmfy_core.responses.ai_response import AIResponse

RetryReason = Literal["error", "empty", "capacity"]

# Status codes that mean "out of capacity" even when the provider's error
# class has no dedicated mapping (e.g. Anthropic's 529 `OverloadedError`).
_CAPACITY_STATUS_CODES = frozenset({429, 503, 529})


def is_retryable_llm_error(error: BaseException) -> bool:
    """Default `LLMRetryPolicy.is_retryable`: failures the SDK does not retry
    but a fresh call can fix.

    - `ModelErrorException` (Bedrock 424: the model failed while processing).
    - `TimeoutException` with `timeout_type=TimeoutType.MODEL` (Bedrock 408:
      the model took too long). Network timeouts (connect/read/write/pool) are
      left to the SDK, which already retried them; retrying them again here
      would multiply long hangs.
    """
    if isinstance(error, ModelErrorException):
        return True
    return isinstance(error, TimeoutException) and error.timeout_type is TimeoutType.MODEL


def is_capacity_error(error: BaseException) -> bool:
    """Default `LLMRetryPolicy.is_capacity_error`: the provider is out of quota
    or capacity (throttling, model not ready, 500/503, 529 overloaded).

    The SDK has already retried these for a few seconds; the policy only keeps
    going when `throttle_wait` is set.
    """
    if isinstance(error, (RateLimitException, ServiceUnavailableException)):
        return True
    return (
        isinstance(error, LLMfyException)
        and error.status_code in _CAPACITY_STATUS_CODES
    )


def is_empty_response(response: AIResponse) -> bool:
    """True when the reply has neither non-blank text nor tool calls."""
    return not (response.content or "").strip() and not response.tool_calls


@dataclass(frozen=True)
class RetryEvent:
    """Passed to `LLMRetryPolicy.on_retry` right before the policy sleeps.

    Attributes:
        model: Model name of the call being retried.
        reason: `"error"` (a retryable error), `"empty"` (an empty reply) or
            `"capacity"` (a capacity error inside the throttle wait).
        attempt: For `"error"`/`"empty"`, the attempt that just failed
            (1-based). For `"capacity"`, the number of capacity retries so far.
        max_attempts: `LLMRetryPolicy.max_attempts`.
        delay: Seconds the policy sleeps before the next attempt.
        error: The exception that triggered the retry (`None` for `"empty"`).
        wait_remaining: For `"capacity"`, seconds left in the throttle wait
            before this sleep; `None` otherwise.
        streaming: True when the retried call is a stream that failed before
            its first chunk.
    """

    model: str
    reason: RetryReason
    attempt: int
    max_attempts: int
    delay: float
    error: BaseException | None = None
    wait_remaining: float | None = None
    streaming: bool = False


@dataclass(frozen=True)
class LLMRetryPolicy:
    """How an `LLMfy` instance retries each model round trip.

    Pass it as `LLMfy(model, retry=LLMRetryPolicy(...))`. The policy belongs to
    the `LLMfy` instance, so two instances sharing one model object can use
    different policies (e.g. a short wait for chat, a long one for background
    jobs).

    Sync methods sleep with `time.sleep` (blocking the calling thread); async
    methods sleep with `asyncio.sleep`.

    Args:
        max_attempts: Total attempts for retryable errors and empty replies,
            including the first call. `1` disables those retries.
        base_delay: Seconds before the second attempt; doubles after each
            retry. Each delay gets random jitter (50-100% of its value).
        max_delay: Upper bound in seconds for one delay.
        throttle_wait: Seconds to keep retrying capacity errors (see
            `is_capacity_error`), counted from the first one. `0` (default)
            gives up as soon as the SDK does. Capacity retries do not use up
            `max_attempts`.
        throttle_base_delay: Seconds before the first capacity retry; doubles
            after each one.
        throttle_max_delay: Upper bound in seconds for one capacity delay. The
            last delay is also clamped to what remains of `throttle_wait`.
        retry_on_empty: Retry replies with neither text nor tool calls.
        on_empty_exhausted: What to do when every attempt came back empty:
            `"return"` the last empty reply (default) or `"raise"`
            `EmptyResponseException`. Ignored when `retry_on_empty` is False.
        retry_stream: Retry `*_stream` calls that fail before their first
            chunk. A stream that already yielded a chunk is never retried, so
            the caller never sees duplicated output.
        is_retryable: Decides whether an error is worth another attempt
            (counted against `max_attempts`). Defaults to
            `is_retryable_llm_error`.
        is_capacity_error: Decides whether an error is a capacity error
            (governed by `throttle_wait`). Defaults to `is_capacity_error`.
        on_retry: Called with a `RetryEvent` before each sleep, for logging or
            metrics. Exceptions it raises propagate to the caller.

    Example:
        ```python
        from llmfy import BedrockConverseModel, LLMfy, LLMRetryPolicy

        llm = BedrockConverseModel(model="amazon.nova-pro-v1:0")
        chat = LLMfy(llm, retry=LLMRetryPolicy(max_attempts=3, throttle_wait=30))
        ```
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 8.0
    throttle_wait: float = 0.0
    throttle_base_delay: float = 2.0
    throttle_max_delay: float = 15.0
    retry_on_empty: bool = True
    on_empty_exhausted: Literal["return", "raise"] = "return"
    retry_stream: bool = True
    is_retryable: Callable[[BaseException], bool] = is_retryable_llm_error
    is_capacity_error: Callable[[BaseException], bool] = is_capacity_error
    on_retry: Callable[[RetryEvent], None] | None = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("LLMRetryPolicy.max_attempts must be at least 1")
        for name in (
            "base_delay",
            "max_delay",
            "throttle_wait",
            "throttle_base_delay",
            "throttle_max_delay",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"LLMRetryPolicy.{name} must not be negative")
        if self.on_empty_exhausted not in ("return", "raise"):
            raise ValueError(
                'LLMRetryPolicy.on_empty_exhausted must be "return" or "raise"'
            )


def backoff_delay(retry: int, base_delay: float, max_delay: float) -> float:
    """Jittered exponential backoff for the `retry`-th retry (1-based).

    Jitter keeps concurrent callers that failed together from retrying in
    lockstep.
    """
    delay = min(max_delay, base_delay * 2 ** (retry - 1))
    return delay * random.uniform(0.5, 1.0)
