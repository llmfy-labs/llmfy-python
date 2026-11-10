"""Transport-level retry and timeout settings handed to a provider SDK.

Pass it to a model constructor (`sdk_retry=SDKRetryConfig(...)`). It tunes
the SDK layer, which retries throttling, 5xx, dropped connections and read
timeouts within seconds; `LLMRetryPolicy` sits on top of it.

Keep `max_attempts` small: every `LLMRetryPolicy` attempt runs up to
`max_attempts` SDK requests, so the two layers multiply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SDKRetryConfig:
    """SDK retry/timeout settings. `None` keeps the SDK's own default.

    Args:
        max_attempts: Total SDK attempts per request, including the first.
            Bedrock: botocore `"standard"` retry mode with this many
            `total_max_attempts`. OpenAI/Anthropic: `max_retries =
            max_attempts - 1`. Google: `HttpRetryOptions(attempts=...)`.
        read_timeout: Seconds to wait for response bytes. Google has a single
            request timeout, which this sets. A hung connection can take
            `max_attempts * read_timeout` before the SDK gives up.
        connect_timeout: Seconds to wait for a connection. Ignored by Google.

    Defaults when a setting is `None` (SDK versions at the time of writing):
    botocore uses `"legacy"` mode with 5 attempts and 60s timeouts unless
    `max_attempts` is set; OpenAI/Anthropic retry twice with a 600s read
    timeout; Google does not retry.
    """

    max_attempts: int | None = None
    read_timeout: float | None = None
    connect_timeout: float | None = None

    def __post_init__(self) -> None:
        if self.max_attempts is not None and self.max_attempts < 1:
            raise ValueError("SDKRetryConfig.max_attempts must be at least 1")
        for name in ("read_timeout", "connect_timeout"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"SDKRetryConfig.{name} must be positive")


def botocore_config(sdk_retry: SDKRetryConfig | None, client_config: Any | None) -> Any | None:
    """Merge `sdk_retry` into an optional botocore `Config` (sdk_retry wins)."""
    if sdk_retry is None:
        return client_config

    from botocore.config import Config

    options: dict[str, Any] = {}
    if sdk_retry.max_attempts is not None:
        options["retries"] = {
            "mode": "standard",
            "total_max_attempts": sdk_retry.max_attempts,
        }
    if sdk_retry.read_timeout is not None:
        options["read_timeout"] = sdk_retry.read_timeout
    if sdk_retry.connect_timeout is not None:
        options["connect_timeout"] = sdk_retry.connect_timeout
    retry_config = Config(**options)
    return client_config.merge(retry_config) if client_config is not None else retry_config


def httpx_sdk_client_options(sdk_retry: SDKRetryConfig | None, sdk_module: Any) -> dict[str, Any]:
    """Client kwargs (`max_retries`, `timeout`) for the `openai`/`anthropic` SDKs.

    `sdk_module` is the imported `openai` or `anthropic` module; its
    `DEFAULT_TIMEOUT` supplies the write/pool timeouts and whichever of
    read/connect is not set.
    """
    if sdk_retry is None:
        return {}
    options: dict[str, Any] = {}
    if sdk_retry.max_attempts is not None:
        options["max_retries"] = sdk_retry.max_attempts - 1
    if sdk_retry.read_timeout is not None or sdk_retry.connect_timeout is not None:
        default = sdk_module.DEFAULT_TIMEOUT
        options["timeout"] = sdk_module.Timeout(
            connect=sdk_retry.connect_timeout
            if sdk_retry.connect_timeout is not None
            else default.connect,
            read=sdk_retry.read_timeout if sdk_retry.read_timeout is not None else default.read,
            write=default.write,
            pool=default.pool,
        )
    return options


def google_http_options(sdk_retry: SDKRetryConfig | None) -> Any | None:
    """`google.genai.types.HttpOptions` for `sdk_retry`, or None."""
    if sdk_retry is None or (
        sdk_retry.max_attempts is None and sdk_retry.read_timeout is None
    ):
        return None

    from google.genai import types

    options: dict[str, Any] = {}
    if sdk_retry.max_attempts is not None:
        options["retry_options"] = types.HttpRetryOptions(attempts=sdk_retry.max_attempts)
    if sdk_retry.read_timeout is not None:
        # google-genai takes the request timeout in milliseconds.
        options["timeout"] = int(sdk_retry.read_timeout * 1000)
    return types.HttpOptions(**options)
