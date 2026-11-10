"""Smoke tests for llmfy/llmfy_core/llms/google/generate/googleai_generate_model.py.

Deep formatting logic is covered by test_googleai_generate_formatter.py; this
covers construction, one successful generate() path, and error mapping.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import errors as google_errors

import llmfy.llmfy_core.llms.google.generate.googleai_generate_model as googleai_model_module
from llmfy.exception.llmfy_exception import (
    LLMfyException,
    RateLimitException,
    TimeoutException,
    TimeoutType,
)
from llmfy.llmfy_core.llms.google.generate.googleai_generate_model import (
    GoogleAIGenerateModel,
)
from llmfy.llmfy_core.model_backend import ModelBackend
from llmfy.llmfy_core.service_provider import ServiceProvider


class FakeGoogleAPIError(google_errors.APIError):
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message


@pytest.fixture
def model(provider_api_keys) -> GoogleAIGenerateModel:
    return GoogleAIGenerateModel(model="gemini-test")


def test_construction_sets_backend_and_provider(model: GoogleAIGenerateModel):
    assert model.backend == ModelBackend.GOOGLE_GENERATE
    assert model.provider == ServiceProvider.GOOGLE


def test_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(LLMfyException, match="GOOGLE_API_KEY"):
        GoogleAIGenerateModel(model="gemini-test")


def test_raises_when_genai_package_missing(provider_api_keys, monkeypatch):
    monkeypatch.setattr(googleai_model_module, "genai", None)
    with pytest.raises(LLMfyException, match="google-genai package is not installed"):
        GoogleAIGenerateModel(model="gemini-test")


def _make_text_response(text: str):
    part = SimpleNamespace(text=text, function_call=None, thought=False)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(candidates=[candidate], text=text)


def test_generate_parses_text_response(model: GoogleAIGenerateModel):
    model.client.models.generate_content = MagicMock(
        return_value=_make_text_response("Hello there")
    )
    result = model.generate(messages=[{"role": "user", "parts": [{"text": "hi"}]}])
    assert result.content == "Hello there"
    assert result.tool_calls is None


def test_generate_parses_function_call_response(model: GoogleAIGenerateModel):
    function_call = SimpleNamespace(
        id="fc_1", name="get_weather", args={"city": "Paris"}
    )
    part = SimpleNamespace(text=None, function_call=function_call, thought=False)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)
    response = SimpleNamespace(candidates=[candidate], text=None)

    model.client.models.generate_content = MagicMock(return_value=response)
    result = model.generate(
        messages=[{"role": "user", "parts": [{"text": "weather?"}]}]
    )
    assert result.tool_calls[0].name == "get_weather"  # type: ignore
    assert result.tool_calls[0].arguments == {"city": "Paris"}  # type: ignore


def test_system_instruction_extracted_from_system_role_message(
    model: GoogleAIGenerateModel,
):
    captured = {}

    def fake_generate_content(**kwargs):
        captured.update(kwargs)
        return _make_text_response("ok")

    model.client.models.generate_content = MagicMock(side_effect=fake_generate_content)
    model.generate(
        messages=[
            {"role": "system", "parts": [{"text": "be nice"}]},
            {"role": "user", "parts": [{"text": "hi"}]},
        ]
    )
    assert captured["config"].system_instruction == "be nice"
    assert all(m.get("role") != "system" for m in captured["contents"])


def test_provider_api_error_is_mapped(model: GoogleAIGenerateModel):
    model.client.models.generate_content = MagicMock(
        side_effect=FakeGoogleAPIError(code=429, message="slow down")
    )
    with pytest.raises(RateLimitException):
        model.generate(messages=[{"role": "user", "parts": [{"text": "hi"}]}])


def test_provider_timeout_is_mapped(model: GoogleAIGenerateModel, httpx_mod):
    # google-genai lets httpx2/httpx timeouts propagate as-is; the model must
    # catch them (not crash on a missing `httpx` import) and map the type.
    model.client.models.generate_content = MagicMock(
        side_effect=httpx_mod.ReadTimeout("read timeout")
    )
    with pytest.raises(TimeoutException) as exc_info:
        model.generate(messages=[{"role": "user", "parts": [{"text": "hi"}]}])
    assert exc_info.value.timeout_type == TimeoutType.READ
    assert exc_info.value.provider == ServiceProvider.GOOGLE


async def test_provider_timeout_is_mapped_async(
    model: GoogleAIGenerateModel, httpx_mod
):
    model.client.aio.models.generate_content = AsyncMock(
        side_effect=httpx_mod.ConnectTimeout("connect timeout")
    )
    with pytest.raises(TimeoutException) as exc_info:
        await model.agenerate(messages=[{"role": "user", "parts": [{"text": "hi"}]}])
    assert exc_info.value.timeout_type == TimeoutType.CONNECT
