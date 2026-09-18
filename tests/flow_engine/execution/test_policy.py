"""Unit tests for llmfy/flow_engine/execution/policy.py (RetryPolicy)."""

from llmfy.exception.llmfy_exception import RateLimitException, TimeoutException
from llmfy.flow_engine.execution.policy import RetryPolicy


class TestDefaults:
    def test_default_is_no_retry(self):
        policy = RetryPolicy()
        assert policy.max_attempts == 1

    def test_default_retry_on_is_exception(self):
        policy = RetryPolicy()
        assert policy.should_retry(ValueError("x"))
        assert policy.should_retry(RuntimeError("x"))

    def test_default_backoff_is_zero(self):
        policy = RetryPolicy()
        assert policy.delay_for_attempt(1) == 0.0
        assert policy.delay_for_attempt(3) == 0.0


class TestShouldRetry:
    def test_matches_configured_exception_types(self):
        policy = RetryPolicy(retry_on=(TimeoutException, RateLimitException))
        assert policy.should_retry(TimeoutException("x"))
        assert policy.should_retry(RateLimitException("x"))

    def test_rejects_unmatched_exception_types(self):
        policy = RetryPolicy(retry_on=(TimeoutException,))
        assert not policy.should_retry(ValueError("x"))

    def test_subclass_matches_registered_base(self):
        policy = RetryPolicy(retry_on=(Exception,))
        assert policy.should_retry(TimeoutException("x"))


class TestBackoff:
    def test_backoff_grows_by_multiplier(self):
        policy = RetryPolicy(backoff_seconds=1.0, backoff_multiplier=2.0)
        assert policy.delay_for_attempt(1) == 1.0
        assert policy.delay_for_attempt(2) == 2.0
        assert policy.delay_for_attempt(3) == 4.0

    def test_zero_backoff_seconds_is_always_zero_delay(self):
        policy = RetryPolicy(backoff_seconds=0.0, backoff_multiplier=5.0)
        assert policy.delay_for_attempt(1) == 0.0
        assert policy.delay_for_attempt(10) == 0.0

    def test_custom_multiplier(self):
        policy = RetryPolicy(backoff_seconds=2.0, backoff_multiplier=1.5)
        assert policy.delay_for_attempt(1) == 2.0
        assert policy.delay_for_attempt(2) == 3.0
