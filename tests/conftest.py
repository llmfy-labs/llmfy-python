"""Shared pytest fixtures for the llmfy test suite.

Scope: everything under `llmfy/`, including `llmfy/flow_engine/` (its own
suite lives under `tests/flow_engine/`, mirroring the package layout).

Mocking strategy: provider SDKs (openai, boto3, anthropic, google-genai) are
all real, lightweight packages installed via the `dev`/`test` dependency
groups. Tests never hit the network — they construct real SDK client objects
(so constructor validation / exception classes are the real ones) and mock
only the network-call method (e.g. `client.chat.completions.create`).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from types import ModuleType
from typing import Any

import pytest

from llmfy.exception._httpx_compat import httpx_modules
from llmfy.llmfy_core.llms.base_ai_model import BaseAIModel
from llmfy.llmfy_core.model_backend import ModelBackend
from llmfy.llmfy_core.responses.ai_response import AIResponse
from llmfy.llmfy_core.service_provider import ServiceProvider


@pytest.fixture
def provider_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dummy credentials for every provider so model constructors don't fail
    on a missing API key. Never real credentials, never used over the network.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-dummy-google-key")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATESTDUMMY")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-dummy-secret")
    monkeypatch.setenv("AWS_BEDROCK_REGION", "us-east-1")


@pytest.fixture(params=httpx_modules(), ids=lambda module: module.__name__)
def httpx_mod(request: pytest.FixtureRequest) -> ModuleType:
    """Each installed httpx-family package (`httpx2` and/or `httpx`), one run per package.

    Current provider SDKs raise `httpx2` exceptions and older ones raise
    `httpx` exceptions; tests taking this fixture prove both are handled.
    """
    return request.param


class FakeAIModel(BaseAIModel):
    """Test double for `BaseAIModel` used to unit-test `LLMfy` without any
    real provider. Every method is driven by an injectable queue/side_effect
    so tests can script multi-round tool-calling scenarios.

    Queued items may be exceptions instead of responses: the call raises
    that exception, which lets tests script failures (e.g. for retry).
    """

    def __init__(
        self,
        backend: ModelBackend = ModelBackend.OPENAI_CHAT,
        provider: ServiceProvider = ServiceProvider.OPENAI,
        responses: list[AIResponse | Exception] | None = None,
        stream_chunks: list[AIResponse] | None = None,
        stream_scripts: list[list[AIResponse | Exception]] | None = None,
    ):
        self.backend = backend
        self.provider = provider
        # Each call to generate()/agenerate() pops the next queued response
        # (or repeats the last one forever if the queue has exactly one item
        # — convenient for single-round tests).
        self._responses = list(responses) if responses else [AIResponse(content="ok")]
        # Every stream call yields `stream_chunks`, unless `stream_scripts`
        # is given: then each stream call consumes the next script (a list of
        # chunks, where an exception is raised at that point in the stream)
        # and streams nothing once the scripts run out.
        self._stream_chunks = list(stream_chunks) if stream_chunks else []
        self._stream_scripts = (
            [list(script) for script in stream_scripts] if stream_scripts is not None else None
        )
        self.calls: list[list[dict[str, Any]]] = []

    def _next_response(self) -> AIResponse:
        response = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(response, Exception):
            raise response
        return response

    def _next_stream(self) -> list[AIResponse | Exception]:
        if self._stream_scripts is None:
            return list(self._stream_chunks)
        return self._stream_scripts.pop(0) if self._stream_scripts else []

    @staticmethod
    def _stream_item(item: AIResponse | Exception) -> AIResponse:
        if isinstance(item, Exception):
            raise item
        return item

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> AIResponse:
        self.calls.append(messages)
        return self._next_response()

    def generate_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> Generator[AIResponse, Any, None]:
        self.calls.append(messages)
        for item in self._next_stream():
            yield self._stream_item(item)

    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> AIResponse:
        self.calls.append(messages)
        return self._next_response()

    async def agenerate_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> AsyncGenerator[AIResponse, Any]:
        self.calls.append(messages)
        for item in self._next_stream():
            yield self._stream_item(item)


@pytest.fixture
def fake_model() -> FakeAIModel:
    return FakeAIModel()
