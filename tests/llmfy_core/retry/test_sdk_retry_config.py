"""SDKRetryConfig: validation and how each model hands it to its SDK client."""

import anthropic
import openai
import pytest
from botocore.config import Config

from llmfy.llmfy_core.llms.anthropic.messages.anthropic_messages_model import (
    AnthropicMessagesModel,
)
from llmfy.llmfy_core.llms.bedrock.converse.bedrock_converse_model import (
    BedrockConverseModel,
)
from llmfy.llmfy_core.llms.google.generate.googleai_generate_model import (
    GoogleAIGenerateModel,
)
from llmfy.llmfy_core.llms.openai.chat.openai_chat_model import OpenAIChatModel
from llmfy.llmfy_core.llms.openai.responses.openai_responses_model import (
    OpenAIResponsesModel,
)
from llmfy.llmfy_core.retry import SDKRetryConfig
from llmfy.llmfy_core.retry.sdk_retry_config import google_http_options


class TestValidation:
    def test_all_none_by_default(self):
        config = SDKRetryConfig()
        assert (config.max_attempts, config.read_timeout, config.connect_timeout) == (
            None,
            None,
            None,
        )

    def test_max_attempts_must_be_positive(self):
        with pytest.raises(ValueError, match="max_attempts"):
            SDKRetryConfig(max_attempts=0)

    @pytest.mark.parametrize("field", ["read_timeout", "connect_timeout"])
    def test_timeouts_must_be_positive(self, field: str):
        with pytest.raises(ValueError, match=field):
            SDKRetryConfig(**{field: 0})


class TestBedrock:
    def test_without_sdk_retry_keeps_botocore_defaults(self, provider_api_keys):
        model = BedrockConverseModel(model="m")
        retries = model.client.meta.config.retries

        assert model._client_config is None
        assert retries.get("mode", "legacy") == "legacy"

    def test_sdk_retry_switches_to_standard_mode(self, provider_api_keys):
        model = BedrockConverseModel(
            model="m",
            sdk_retry=SDKRetryConfig(max_attempts=3, read_timeout=45, connect_timeout=5),
        )
        config = model.client.meta.config

        assert config.retries == {"mode": "standard", "total_max_attempts": 3}
        assert config.read_timeout == 45
        assert config.connect_timeout == 5

    def test_client_config_is_kept_and_sdk_retry_wins(self, provider_api_keys):
        model = BedrockConverseModel(
            model="m",
            sdk_retry=SDKRetryConfig(max_attempts=2),
            client_config=Config(max_pool_connections=32, retries={"max_attempts": 9}),
        )
        config = model.client.meta.config

        assert config.max_pool_connections == 32
        assert config.retries == {"mode": "standard", "total_max_attempts": 2}

    def test_client_config_alone_is_passed_through(self, provider_api_keys):
        model = BedrockConverseModel(model="m", client_config=Config(max_pool_connections=24))

        assert model.client.meta.config.max_pool_connections == 24

    async def test_async_client_gets_the_same_config(self, provider_api_keys):
        pytest.importorskip("aioboto3")
        model = BedrockConverseModel(
            model="m",
            sdk_retry=SDKRetryConfig(max_attempts=3, read_timeout=45),
            client_config=Config(max_pool_connections=32),
        )

        async with model._aioboto3_session.client(
            "bedrock-runtime", config=model._client_config
        ) as client:
            config = client.meta.config

        assert config.retries == {"mode": "standard", "total_max_attempts": 3}
        assert config.read_timeout == 45
        assert config.max_pool_connections == 32


@pytest.mark.parametrize(
    ("model_cls", "sdk"),
    [
        (OpenAIChatModel, openai),
        (OpenAIResponsesModel, openai),
        (AnthropicMessagesModel, anthropic),
    ],
    ids=["openai-chat", "openai-responses", "anthropic"],
)
class TestHttpxSDKs:
    def test_without_sdk_retry_keeps_sdk_defaults(self, provider_api_keys, model_cls, sdk):
        model = model_cls(model="m")

        for client in (model.client, model.async_client):
            assert client.max_retries == sdk.DEFAULT_MAX_RETRIES
            assert client.timeout == sdk.DEFAULT_TIMEOUT

    def test_sdk_retry_sets_retries_and_timeouts(self, provider_api_keys, model_cls, sdk):
        model = model_cls(
            model="m",
            sdk_retry=SDKRetryConfig(max_attempts=3, read_timeout=45, connect_timeout=4),
        )

        for client in (model.client, model.async_client):
            assert client.max_retries == 2
            assert client.timeout.read == 45
            assert client.timeout.connect == 4
            assert client.timeout.write == sdk.DEFAULT_TIMEOUT.write
            assert client.timeout.pool == sdk.DEFAULT_TIMEOUT.pool

    def test_one_attempt_means_no_sdk_retries(self, provider_api_keys, model_cls, sdk):
        model = model_cls(model="m", sdk_retry=SDKRetryConfig(max_attempts=1))

        assert model.client.max_retries == 0
        assert model.client.timeout == sdk.DEFAULT_TIMEOUT

    def test_read_timeout_alone_keeps_the_default_connect_timeout(
        self, provider_api_keys, model_cls, sdk
    ):
        model = model_cls(model="m", sdk_retry=SDKRetryConfig(read_timeout=30))

        assert model.client.max_retries == sdk.DEFAULT_MAX_RETRIES
        assert model.client.timeout.read == 30
        assert model.client.timeout.connect == sdk.DEFAULT_TIMEOUT.connect


class TestGoogle:
    def test_without_sdk_retry_builds_no_http_options(self):
        assert google_http_options(None) is None
        assert google_http_options(SDKRetryConfig(connect_timeout=3)) is None

    def test_sdk_retry_maps_to_http_options(self):
        options = google_http_options(SDKRetryConfig(max_attempts=4, read_timeout=12.5))

        assert options.retry_options.attempts == 4
        assert options.timeout == 12500

    def test_model_client_uses_the_http_options(self, provider_api_keys):
        model = GoogleAIGenerateModel(
            model="gemini-2.0-flash",
            sdk_retry=SDKRetryConfig(max_attempts=4, read_timeout=12.5),
        )
        http_options = model.client._api_client._http_options

        assert http_options.retry_options.attempts == 4
        assert http_options.timeout == 12500
