"""Fixtures for the retry tests: a fake clock for the retry runner.

Scripted model failures come from `tests.conftest.FakeAIModel`, whose
queues accept exceptions.
"""

from __future__ import annotations

import pytest

from llmfy.llmfy_core.retry import retry_runner


class FakeClock:
    """Replaces the retry runner's clock and sleeps: sleeping advances time."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    async def async_sleep(self, seconds: float) -> None:
        self.sleep(seconds)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(retry_runner, "_monotonic", fake.monotonic)
    monkeypatch.setattr(retry_runner, "_sleep", fake.sleep)
    monkeypatch.setattr(retry_runner, "_async_sleep", fake.async_sleep)
    return fake
