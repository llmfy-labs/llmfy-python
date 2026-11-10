"""Runs one model round trip (or one stream) under an `LLMRetryPolicy`.

Used by `LLMfy`; each function takes a zero-argument callable that performs a
single fresh attempt, so request payloads are rebuilt for every attempt.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    Awaitable,
    Callable,
    Generator,
    Iterable,
)
from typing import Any

from llmfy.exception.llmfy_exception import EmptyResponseException
from llmfy.llmfy_core.responses.ai_response import AIResponse
from llmfy.llmfy_core.retry.llm_retry_policy import (
    LLMRetryPolicy,
    RetryEvent,
    RetryReason,
    backoff_delay,
    is_empty_response,
)

logger = logging.getLogger(__name__)

# Indirection so tests can replace the clock and sleeps without patching the
# global `time`/`asyncio` modules.
_monotonic = time.monotonic
_sleep = time.sleep
_async_sleep = asyncio.sleep


class _RetryState:
    """Attempt and throttle-wait bookkeeping for one call."""

    def __init__(self, policy: LLMRetryPolicy, model: str, streaming: bool) -> None:
        self.policy = policy
        self.model = model
        self.streaming = streaming
        self.attempt = 1
        self.throttle_retry = 0
        self.throttle_deadline: float | None = None

    def delay_after_error(self, error: BaseException) -> float | None:
        """Seconds to sleep before retrying `error`, or None to give up."""
        policy = self.policy
        if policy.throttle_wait > 0 and policy.is_capacity_error(error):
            now = _monotonic()
            if self.throttle_deadline is None:
                self.throttle_deadline = now + policy.throttle_wait
            remaining = self.throttle_deadline - now
            if remaining <= 0:
                return None
            self.throttle_retry += 1
            delay = min(
                remaining,
                backoff_delay(
                    self.throttle_retry,
                    policy.throttle_base_delay,
                    policy.throttle_max_delay,
                ),
            )
            self._notify("capacity", self.throttle_retry, delay, error, remaining)
            return delay
        if self.attempt >= policy.max_attempts or not policy.is_retryable(error):
            return None
        return self._next_attempt("error", error)

    def delay_after_empty(self) -> float | None:
        """Seconds to sleep before retrying an empty reply, or None to stop."""
        if self.attempt >= self.policy.max_attempts:
            return None
        return self._next_attempt("empty", None)

    def _next_attempt(self, reason: RetryReason, error: BaseException | None) -> float:
        policy = self.policy
        delay = backoff_delay(self.attempt, policy.base_delay, policy.max_delay)
        self._notify(reason, self.attempt, delay, error, None)
        self.attempt += 1
        return delay

    def _notify(
        self,
        reason: RetryReason,
        attempt: int,
        delay: float,
        error: BaseException | None,
        wait_remaining: float | None,
    ) -> None:
        event = RetryEvent(
            model=self.model,
            reason=reason,
            attempt=attempt,
            max_attempts=self.policy.max_attempts,
            delay=delay,
            error=error,
            wait_remaining=wait_remaining,
            streaming=self.streaming,
        )
        if reason == "capacity":
            logger.warning(
                "[LLM Retry] %s out of capacity: %r; retry %d in %.1fs "
                "(%.0fs of %.0fs wait left)",
                event.model,
                error,
                attempt,
                delay,
                wait_remaining,
                self.policy.throttle_wait,
            )
        else:
            logger.warning(
                "[LLM Retry] %s attempt %d/%d %s; retrying in %.1fs",
                event.model,
                attempt,
                event.max_attempts,
                "returned an empty reply" if reason == "empty" else f"failed: {error!r}",
                delay,
            )
        if self.policy.on_retry is not None:
            self.policy.on_retry(event)


def _empty_exhausted(policy: LLMRetryPolicy, response: AIResponse, model: str) -> AIResponse:
    if policy.on_empty_exhausted == "raise":
        raise EmptyResponseException(
            f"{model} returned an empty reply on all {policy.max_attempts} attempts",
            response=response,
        )
    return response


def generate_with_retry(
    policy: LLMRetryPolicy, model: str, call: Callable[[], AIResponse]
) -> AIResponse:
    """Run `call` until it returns a usable reply or the policy gives up."""
    state = _RetryState(policy, model, streaming=False)
    while True:
        try:
            response = call()
        except Exception as error:
            delay = state.delay_after_error(error)
            if delay is None:
                raise
            _sleep(delay)
            continue
        if policy.retry_on_empty and is_empty_response(response):
            delay = state.delay_after_empty()
            if delay is None:
                return _empty_exhausted(policy, response, model)
            _sleep(delay)
            continue
        return response


async def agenerate_with_retry(
    policy: LLMRetryPolicy, model: str, call: Callable[[], Awaitable[AIResponse]]
) -> AIResponse:
    """Async version of `generate_with_retry`; sleeps with `asyncio.sleep`."""
    state = _RetryState(policy, model, streaming=False)
    while True:
        try:
            response = await call()
        except Exception as error:
            delay = state.delay_after_error(error)
            if delay is None:
                raise
            await _async_sleep(delay)
            continue
        if policy.retry_on_empty and is_empty_response(response):
            delay = state.delay_after_empty()
            if delay is None:
                return _empty_exhausted(policy, response, model)
            await _async_sleep(delay)
            continue
        return response


def stream_with_retry(
    policy: LLMRetryPolicy, model: str, open_stream: Callable[[], Iterable[Any]]
) -> Generator[Any, None, None]:
    """Yield from a fresh stream, retrying only while nothing has been yielded.

    Once a chunk reaches the caller, any later error is raised as-is: a retry
    would repeat output the caller already consumed. Empty replies are not
    retried for streams.
    """
    if not policy.retry_stream:
        yield from open_stream()
        return
    state = _RetryState(policy, model, streaming=True)
    while True:
        yielded = False
        try:
            for chunk in open_stream():
                yielded = True
                yield chunk
            return
        except Exception as error:
            if yielded:
                raise
            delay = state.delay_after_error(error)
            if delay is None:
                raise
            _sleep(delay)


async def astream_with_retry(
    policy: LLMRetryPolicy, model: str, open_stream: Callable[[], AsyncIterable[Any]]
) -> AsyncGenerator[Any, None]:
    """Async version of `stream_with_retry`; sleeps with `asyncio.sleep`."""
    if not policy.retry_stream:
        async for chunk in open_stream():
            yield chunk
        return
    state = _RetryState(policy, model, streaming=True)
    while True:
        yielded = False
        try:
            async for chunk in open_stream():
                yielded = True
                yield chunk
            return
        except Exception as error:
            if yielded:
                raise
            delay = state.delay_after_error(error)
            if delay is None:
                raise
            await _async_sleep(delay)
