from dataclasses import dataclass, field


@dataclass
class RetryPolicy:
    """Retry behavior for a single node's execution.

    `max_attempts=1` (the default) means no retry — a node either succeeds
    or its exception propagates immediately, matching the engine's
    pre-existing behavior. Set `max_attempts` above 1 to retry transient
    failures (a flaky LLM/tool call) with exponential backoff.
    """

    max_attempts: int = 1
    backoff_seconds: float = 0.0
    backoff_multiplier: float = 2.0
    retry_on: tuple[type[BaseException], ...] = field(default_factory=lambda: (Exception,))

    def delay_for_attempt(self, attempt: int) -> float:
        """Backoff delay (seconds) before the given attempt number (1-indexed retry count)."""
        if self.backoff_seconds <= 0:
            return 0.0
        return self.backoff_seconds * (self.backoff_multiplier ** (attempt - 1))

    def should_retry(self, exc: BaseException) -> bool:
        return isinstance(exc, self.retry_on)
