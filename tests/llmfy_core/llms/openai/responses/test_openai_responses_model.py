"""Smoke tests for llmfy/llmfy_core/llms/openai/responses/openai_responses_model.py.

Deep formatting logic is covered by test_openai_responses_formatter.py; this
covers construction, one successful generate() path, and error mapping —
proving the model correctly wires into the shared exception-handling and
formatter machinery.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import openai
import pytest

from llmfy.exception._httpx_compat import httpx_modules
from llmfy.exception.llmfy_exception import LLMfyException, RateLimitException
from llmfy.llmfy_core.llms.openai.responses.openai_responses_model import (
    OpenAIResponsesModel,
)
from llmfy.llmfy_core.model_backend import ModelBackend
from llmfy.llmfy_core.service_provider import ServiceProvider

# The HTTP package the provider SDK builds requests/responses with: `httpx2` on
# current SDK releases, `httpx` on older ones.
httpx = httpx_modules()[0]


@pytest.fixture
def model(provider_api_keys) -> OpenAIResponsesModel:
    return OpenAIResponsesModel(model="gpt-responses-test")


def test_construction_sets_backend_and_provider(model: OpenAIResponsesModel):
    assert model.backend == ModelBackend.OPENAI_RESPONSES
    assert model.provider == ServiceProvider.OPENAI


def test_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMfyException, match="OPENAI_API_KEY"):
        OpenAIResponsesModel(model="gpt-responses-test")


def test_generate_parses_message_output_item(model: OpenAIResponsesModel):
    output_text = SimpleNamespace(type="output_text", text="Hello there")
    message_item = SimpleNamespace(type="message", content=[output_text])
    response = SimpleNamespace(id="resp_1", output=[message_item])
    model.client.responses.create = MagicMock(return_value=response)

    result = model.generate(
        messages=[{"type": "message", "role": "user", "content": []}]
    )
    assert result.content == "Hello there"
    assert result.tool_calls is None


def test_generate_parses_function_call_output_item(model: OpenAIResponsesModel):
    call_item = SimpleNamespace(
        type="function_call",
        call_id="call_1",
        name="get_weather",
        arguments='{"city": "Paris"}',
    )
    response = SimpleNamespace(id="resp_1", output=[call_item])
    model.client.responses.create = MagicMock(return_value=response)

    result = model.generate(
        messages=[{"type": "message", "role": "user", "content": []}]
    )
    assert result.content is None
    assert result.tool_calls[0].name == "get_weather" # type: ignore
    assert result.tool_calls[0].arguments == {"city": "Paris"} # type: ignore


def test_provider_api_error_is_mapped(model: OpenAIResponsesModel):
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(status_code=429, request=request)
    model.client.responses.create = MagicMock(
        side_effect=openai.RateLimitError("slow down", response=response, body=None)
    )
    with pytest.raises(RateLimitException):
        model.generate(messages=[{"type": "message", "role": "user", "content": []}])
