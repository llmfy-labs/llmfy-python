"""Smoke tests for llmfy/llmfy_core/llms/anthropic/messages/anthropic_messages_model.py.

Deep formatting logic is covered by test_anthropic_messages_formatter.py;
this covers construction, one successful generate() path, and error mapping.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import pytest

import llmfy.llmfy_core.llms.anthropic.messages.anthropic_messages_model as anthropic_model_module
from llmfy.exception._httpx_compat import httpx_modules
from llmfy.exception.llmfy_exception import (
    LLMfyException,
    RateLimitException,
    TimeoutException,
    TimeoutType,
)
from llmfy.llmfy_core.llms.anthropic.messages.anthropic_messages_model import (
    AnthropicMessagesModel,
)
from llmfy.llmfy_core.model_backend import ModelBackend
from llmfy.llmfy_core.service_provider import ServiceProvider

# The HTTP package the provider SDK builds requests/responses with: `httpx2` on
# current SDK releases, `httpx` on older ones.
httpx = httpx_modules()[0]


@pytest.fixture
def model(provider_api_keys) -> AnthropicMessagesModel:
    return AnthropicMessagesModel(model="claude-sonnet-5-test")


def test_construction_sets_backend_and_provider(model: AnthropicMessagesModel):
    assert model.backend == ModelBackend.ANTHROPIC_MESSAGES
    assert model.provider == ServiceProvider.ANTHROPIC


def test_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMfyException, match="ANTHROPIC_API_KEY"):
        AnthropicMessagesModel(model="claude-sonnet-5-test")


def test_raises_when_anthropic_package_missing(provider_api_keys, monkeypatch):
    monkeypatch.setattr(anthropic_model_module, "anthropic", None)
    with pytest.raises(LLMfyException, match="anthropic package is not installed"):
        AnthropicMessagesModel(model="claude-sonnet-5-test")


def test_generate_parses_text_response(model: AnthropicMessagesModel):
    text_block = SimpleNamespace(type="text", text="Hello there")
    response = SimpleNamespace(id="msg_1", stop_reason="end_turn", content=[text_block])
    model.client.messages.create = MagicMock(return_value=response)

    result = model.generate(
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    )
    assert result.content == "Hello there"
    assert result.tool_calls is None


def test_generate_parses_tool_use_response(model: AnthropicMessagesModel):
    tool_use_block = SimpleNamespace(
        type="tool_use", id="toolu_1", name="get_weather", input={"city": "Paris"}
    )
    response = SimpleNamespace(
        id="msg_1", stop_reason="tool_use", content=[tool_use_block]
    )
    model.client.messages.create = MagicMock(return_value=response)

    result = model.generate(
        messages=[{"role": "user", "content": [{"type": "text", "text": "weather?"}]}]
    )
    assert result.content is None
    assert result.tool_calls[0].name == "get_weather"  # type: ignore
    assert result.tool_calls[0].arguments == {"city": "Paris"}  # type: ignore


def test_system_message_hoisted_out_of_messages_array(model: AnthropicMessagesModel):
    captured = {}

    def fake_create(**params):
        captured.update(params)
        return SimpleNamespace(
            id="msg_1",
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="ok")],
        )

    model.client.messages.create = MagicMock(side_effect=fake_create)
    model.generate(
        messages=[
            {"role": "system", "content": [{"type": "text", "text": "be nice"}]},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ]
    )
    assert captured["system"] == [{"type": "text", "text": "be nice"}]
    assert all(m["role"] != "system" for m in captured["messages"])


def test_provider_api_error_is_mapped(model: AnthropicMessagesModel):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code=429, request=request)
    model.client.messages.create = MagicMock(
        side_effect=anthropic.RateLimitError("slow down", response=response, body=None)
    )
    with pytest.raises(RateLimitException):
        model.generate(
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        )


def test_provider_timeout_maps_timeout_type_from_cause(
    model: AnthropicMessagesModel, httpx_mod
):
    # Mirrors how the SDK raises it: `APITimeoutError(...) from <httpx2/httpx timeout>`.
    request = httpx_mod.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.APITimeoutError(request=request)
    error.__cause__ = httpx_mod.ConnectTimeout("connect timeout", request=request)
    model.client.messages.create = MagicMock(side_effect=error)
    with pytest.raises(TimeoutException) as exc_info:
        model.generate(
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        )
    assert exc_info.value.timeout_type == TimeoutType.CONNECT
    assert exc_info.value.provider == ServiceProvider.ANTHROPIC
