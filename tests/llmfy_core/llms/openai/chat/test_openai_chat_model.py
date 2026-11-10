"""Unit tests for llmfy/llmfy_core/llms/openai/chat/openai_chat_model.py.

Mocking strategy: real `openai` SDK client objects are constructed (so
constructor validation is real), but the network-call method
(`client.chat.completions.create`) is mocked at the boundary — no real API
key, no network access. Provider errors are tested with real `openai`
exception classes so the model -> `handle_openai_error` wiring is proven
end-to-end, not just the handler in isolation (see
tests/exception/test_exception_handler.py for that).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest

import llmfy.llmfy_core.llms.openai.chat.openai_chat_model as openai_chat_model_module
from llmfy.exception._httpx_compat import httpx_modules
from llmfy.exception.llmfy_exception import (
    LLMfyException,
    RateLimitException,
    TimeoutException,
    TimeoutType,
)
from llmfy.llmfy_core.llms.openai.chat.openai_chat_model import OpenAIChatModel
from llmfy.llmfy_core.model_backend import ModelBackend
from llmfy.llmfy_core.service_provider import ServiceProvider

# The HTTP package the provider SDK builds requests/responses with: `httpx2` on
# current SDK releases, `httpx` on older ones.
httpx = httpx_modules()[0]


def make_completion(content=None, tool_calls=None, response_id="resp_1"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(id=response_id, choices=[choice], usage=None)


def make_tool_call_obj(call_id: str, name: str, arguments_json: str):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments_json)
    )


def make_rate_limit_error() -> openai.RateLimitError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request)
    return openai.RateLimitError("Rate limit exceeded", response=response, body=None)


@pytest.fixture
def model(provider_api_keys) -> OpenAIChatModel:
    return OpenAIChatModel(model="gpt-test")


class TestConstruction:
    def test_sets_backend_and_provider(self, model: OpenAIChatModel):
        assert model.backend == ModelBackend.OPENAI_CHAT
        assert model.provider == ServiceProvider.OPENAI
        assert model.model_name == "gpt-test"

    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(LLMfyException, match="OPENAI_API_KEY"):
            OpenAIChatModel(model="gpt-test")

    def test_raises_when_openai_package_missing(self, provider_api_keys, monkeypatch):
        monkeypatch.setattr(openai_chat_model_module, "openai", None)
        with pytest.raises(LLMfyException, match="openai package is not installed"):
            OpenAIChatModel(model="gpt-test")

    def test_explicit_api_key_overrides_env(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        model = OpenAIChatModel(model="gpt-test", api_key="sk-explicit")
        assert model.client.api_key == "sk-explicit"


class TestGenerate:
    def test_returns_content_when_no_tool_calls(self, model: OpenAIChatModel):
        model.client.chat.completions.create = MagicMock(
            return_value=make_completion(content="Hello there")
        )
        response = model.generate(messages=[{"role": "user", "content": "hi"}])
        assert response.content == "Hello there"
        assert response.tool_calls is None

    def test_returns_tool_calls_and_no_content(self, model: OpenAIChatModel):
        tool_call = make_tool_call_obj("call_1", "get_weather", '{"city": "Paris"}')
        model.client.chat.completions.create = MagicMock(
            return_value=make_completion(tool_calls=[tool_call], response_id="resp_42")
        )
        response = model.generate(messages=[{"role": "user", "content": "weather?"}])
        assert response.content is None
        assert len(response.tool_calls) == 1  # type: ignore
        assert response.tool_calls[0].name == "get_weather"  # type: ignore
        assert response.tool_calls[0].arguments == {"city": "Paris"}  # type: ignore
        assert response.tool_calls[0].request_call_id == "resp_42"  # type: ignore

    def test_provider_api_error_is_mapped_via_handle_openai_error(
        self, model: OpenAIChatModel
    ):
        model.client.chat.completions.create = MagicMock(
            side_effect=make_rate_limit_error()
        )
        with pytest.raises(RateLimitException) as exc_info:
            model.generate(messages=[{"role": "user", "content": "hi"}])
        assert exc_info.value.provider == ServiceProvider.OPENAI

    def test_provider_timeout_maps_timeout_type_from_cause(
        self, model: OpenAIChatModel, httpx_mod
    ):
        # Mirrors how the SDK raises it: `APITimeoutError(...) from <httpx2/httpx timeout>`.
        request = httpx_mod.Request("POST", "https://api.openai.com/v1/chat/completions")
        error = openai.APITimeoutError(request=request)
        error.__cause__ = httpx_mod.ReadTimeout("read timeout", request=request)
        model.client.chat.completions.create = MagicMock(side_effect=error)
        with pytest.raises(TimeoutException) as exc_info:
            model.generate(messages=[{"role": "user", "content": "hi"}])
        assert exc_info.value.timeout_type == TimeoutType.READ
        assert exc_info.value.provider == ServiceProvider.OPENAI

    def test_generic_exception_is_wrapped_as_llmfy_exception(
        self, model: OpenAIChatModel
    ):
        model.client.chat.completions.create = MagicMock(
            side_effect=RuntimeError("network down")
        )
        with pytest.raises(LLMfyException) as exc_info:
            model.generate(messages=[{"role": "user", "content": "hi"}])
        assert not isinstance(exc_info.value, RateLimitException)
        assert "network down" in exc_info.value.message

    def test_tools_param_adds_tool_choice_auto(self, model: OpenAIChatModel):
        captured = {}

        def fake_create(**params):
            captured.update(params)
            return make_completion(content="ok")

        model.client.chat.completions.create = MagicMock(side_effect=fake_create)
        model.generate(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"name": "get_weather", "parameters": {}}],
        )
        assert captured["tool_choice"] == "auto"
        assert captured["tools"][0]["type"] == "function"

    def test_max_tokens_omitted_from_request_when_none(self, model: OpenAIChatModel):
        captured = {}

        def fake_create(**params):
            captured.update(params)
            return make_completion(content="ok")

        model.client.chat.completions.create = MagicMock(side_effect=fake_create)
        model.generate(messages=[{"role": "user", "content": "hi"}])
        assert "max_tokens" not in captured


class TestAsyncGenerate:
    @pytest.mark.asyncio
    async def test_returns_content(self, model: OpenAIChatModel):
        model.async_client.chat.completions.create = AsyncMock(
            return_value=make_completion(content="async hello")
        )
        response = await model.agenerate(messages=[{"role": "user", "content": "hi"}])
        assert response.content == "async hello"

    @pytest.mark.asyncio
    async def test_provider_api_error_is_mapped(self, model: OpenAIChatModel):
        model.async_client.chat.completions.create = AsyncMock(
            side_effect=make_rate_limit_error()
        )
        with pytest.raises(RateLimitException):
            await model.agenerate(messages=[{"role": "user", "content": "hi"}])


class TestGenerateStream:
    def test_content_chunks_are_yielded(self, model: OpenAIChatModel):
        def make_chunk(content):
            delta = SimpleNamespace(content=content, tool_calls=None)
            return SimpleNamespace(
                id="c1", choices=[SimpleNamespace(delta=delta)], usage=None
            )

        model.client.chat.completions.create = MagicMock(
            return_value=[make_chunk("Hel"), make_chunk("lo")]
        )
        chunks = list(
            model.generate_stream(messages=[{"role": "user", "content": "hi"}])
        )
        assert [c.content for c in chunks] == ["Hel", "lo"]

    def test_chunk_with_no_choices_is_skipped(self, model: OpenAIChatModel):
        # The trailing usage-only chunk (stream_options include_usage) has an
        # empty `choices` list and must not be yielded as an AIResponse.
        no_choice_chunk = SimpleNamespace(id="c1", choices=[], usage=SimpleNamespace())
        delta = SimpleNamespace(content="hi", tool_calls=None)
        content_chunk = SimpleNamespace(
            id="c1", choices=[SimpleNamespace(delta=delta)], usage=None
        )

        model.client.chat.completions.create = MagicMock(
            return_value=[content_chunk, no_choice_chunk]
        )
        chunks = list(
            model.generate_stream(messages=[{"role": "user", "content": "hi"}])
        )
        assert len(chunks) == 1

    def test_two_parallel_tool_calls_accumulate_independently_by_index(
        self, model: OpenAIChatModel
    ):
        # Interleaved deltas for index 0 and index 1 — correctness here means
        # each tool call's arguments are assembled from only its own chunks.
        def delta_chunk(tool_call_deltas):
            delta = SimpleNamespace(content=None, tool_calls=tool_call_deltas)
            return SimpleNamespace(
                id="c1", choices=[SimpleNamespace(delta=delta)], usage=None
            )

        def tc_delta(index, id=None, name=None, arguments=""):
            function = SimpleNamespace(name=name, arguments=arguments)
            return SimpleNamespace(index=index, id=id, function=function)

        chunks_in = [
            delta_chunk([tc_delta(0, id="call_a", name="fn_a", arguments='{"x"')]),
            delta_chunk([tc_delta(1, id="call_b", name="fn_b", arguments='{"y"')]),
            delta_chunk([tc_delta(0, arguments=": 1}")]),
            delta_chunk([tc_delta(1, arguments=": 2}")]),
        ]
        model.client.chat.completions.create = MagicMock(return_value=chunks_in)

        results = list(
            model.generate_stream(messages=[{"role": "user", "content": "hi"}])
        )
        completed_calls = [tc for r in results if r.tool_calls for tc in r.tool_calls]
        by_name = {tc.name: tc.arguments for tc in completed_calls}
        assert by_name == {"fn_a": {"x": 1}, "fn_b": {"y": 2}}

    def test_provider_api_error_is_mapped(self, model: OpenAIChatModel):
        model.client.chat.completions.create = MagicMock(
            side_effect=make_rate_limit_error()
        )
        with pytest.raises(RateLimitException):
            list(model.generate_stream(messages=[{"role": "user", "content": "hi"}]))
