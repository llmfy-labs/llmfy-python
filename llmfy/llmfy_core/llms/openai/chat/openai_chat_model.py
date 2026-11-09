try:
    import openai
except ImportError:
    openai = None

import json
import os
from collections.abc import AsyncGenerator
from typing import Any

from llmfy.exception.llmfy_exception import LLMfyException
from llmfy.llmfy_core.llms.base_ai_model import BaseAIModel
from llmfy.llmfy_core.llms.openai.chat.openai_chat_config import OpenAIChatConfig
from llmfy.llmfy_core.messages.tool_call import ToolCall
from llmfy.llmfy_core.model_backend import ModelBackend
from llmfy.llmfy_core.responses.ai_response import AIResponse
from llmfy.llmfy_core.retry.sdk_retry_config import (
    SDKRetryConfig,
    httpx_sdk_client_options,
)
from llmfy.llmfy_core.service_provider import ServiceProvider


class OpenAIChatModel(BaseAIModel):
    """
    OpenAIChatModel class.

    Example:
    ```python
    # Configuration
    config = OpenAIChatConfig(
            temperature=0.7
    )
    llm = OpenAIChatModel(model="gpt-4o-mini", config=config)
    ...
    ```
    """

    def __init__(
        self,
        model: str,
        config: OpenAIChatConfig | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        sdk_retry: SDKRetryConfig | None = None,
    ):
        """
        OpenAIChatModel

        Args:
            model (str): Model ID
            config (OpenAIChatConfig, optional): Configuration. Defaults to OpenAIChatConfig().
            api_key (str, optional): OpenAI API key. Defaults to the `OPENAI_API_KEY`
                environment variable if not provided.
            base_url (str, optional): Base URL for the OpenAI API. Defaults to None,
                which uses the OpenAI SDK's default base URL.
            default_headers (dict[str, str], optional): Extra HTTP headers sent on every
                request, passed straight through to the `openai` SDK client. Needed for
                compatible endpoints that require headers beyond `Authorization` — e.g.
                Bedrock Mantle's Project-scoping headers.
            sdk_retry (SDKRetryConfig, optional): `openai` SDK retry/timeout settings
                (`max_retries`, `timeout`) for both the sync and async client.
                Defaults to None (SDK defaults).
        """
        config = config if config is not None else OpenAIChatConfig()
        if openai is None:
            raise LLMfyException(
                'openai package is not installed. Install it using `pip install "llmy[openai]"`'
            )
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMfyException(
                "Please provide `OPENAI_API_KEY` on your environment or pass `api_key`!"
            )

        sdk_options = httpx_sdk_client_options(sdk_retry, openai)
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            **sdk_options,
        )
        # Native async client for `agenerate` — same credentials/config, no
        # separate setup needed. Cheap to construct (no connection opened
        # until the first request), so built eagerly alongside `self.client`
        # rather than lazily on first async use.
        self.async_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            **sdk_options,
        )
        self.backend = ModelBackend.OPENAI_CHAT
        self.provider = ServiceProvider.OPENAI
        self.model_name = model
        self.config = config

    def __call_openai(self, params: dict[str, Any]):
        # Import the decorator when the method is first defined/called
        import openai

        from llmfy.exception.exception_handler import handle_openai_error
        from llmfy.llmfy_core.llms.openai.chat.openai_chat_usage import (
            track_openai_usage,
        )

        @track_openai_usage
        def _call_openai_impl(params: dict[str, Any]):
            try:
                response = self.client.chat.completions.create(**params)
                return response
            except openai.APIError as e:
                raise handle_openai_error(e) from e
            # Any non-openai.APIError exceptions will naturally propagate up the call stack.

        return _call_openai_impl(params)

    def __call_openai_async(self, params: dict[str, Any]):
        # Async counterpart of `__call_openai`, used by `agenerate`. Same
        # `track_openai_usage` decorator — it dispatches on whether the
        # wrapped function is a coroutine function.
        import openai

        from llmfy.exception.exception_handler import handle_openai_error
        from llmfy.llmfy_core.llms.openai.chat.openai_chat_usage import (
            track_openai_usage,
        )

        @track_openai_usage
        async def _call_openai_impl_async(params: dict[str, Any]):
            try:
                response = await self.async_client.chat.completions.create(**params)
                return response
            except openai.APIError as e:
                raise handle_openai_error(e) from e
            # Any non-openai.APIError exceptions will naturally propagate up the call stack.

        return _call_openai_impl_async(params)

    def __call_stream_openai(self, params: dict[str, Any]):
        # Import the decorator when the method is first defined/called
        import openai

        from llmfy.exception.exception_handler import handle_openai_error
        from llmfy.llmfy_core.llms.openai.chat.openai_chat_usage import (
            track_openai_stream_usage,
        )

        @track_openai_stream_usage
        def __call_stream_openai_impl(params: dict[str, Any]):
            try:
                params["stream"] = True
                params["stream_options"] = {"include_usage": True}
                return self.client.chat.completions.create(**params)
            except openai.APIError as e:
                raise handle_openai_error(e) from e
            # Any non-openai.APIError exceptions will naturally propagate up the call stack.

        return __call_stream_openai_impl(params)

    def __call_stream_openai_async(self, params: dict[str, Any]):
        # Async counterpart of `__call_stream_openai`, used by
        # `agenerate_stream`. Unlike Bedrock's aioboto3 client, OpenAI's
        # AsyncStream has no connection-closing-on-context-exit constraint —
        # it's written as an async generator anyway (rather than "await once,
        # return the stream object") purely so usage-tracking can forward
        # chunks in a single pass instead of needing an async `tee`.
        import openai

        from llmfy.exception.exception_handler import handle_openai_error
        from llmfy.llmfy_core.llms.openai.chat.openai_chat_usage import (
            track_openai_stream_usage_async,
        )

        @track_openai_stream_usage_async
        async def _call_stream_openai_impl_async(params: dict[str, Any]):
            try:
                params["stream"] = True
                params["stream_options"] = {"include_usage": True}
                stream = await self.async_client.chat.completions.create(**params)
                async for chunk in stream:
                    yield chunk
            except openai.APIError as e:
                raise handle_openai_error(e) from e

        return _call_stream_openai_impl_async(params)

    def __build_params(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        **kwargs,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            "stream": False,
            **kwargs,
        }
        # Omitted entirely (not sent as null) when None — some models
        # reject these params outright rather than accepting a default
        # (e.g. o4-mini 400s on an explicit `max_tokens: null`).
        if self.config.max_tokens is not None:
            params["max_tokens"] = self.config.max_tokens
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            params["top_p"] = self.config.top_p

        if self.config.thinking.enabled:
            params["reasoning_effort"] = self.config.thinking.effort or "medium"

        if tools:
            params["tools"] = [
                {"type": "function", "function": tool} for tool in tools
            ]
            params["tool_choice"] = "auto"

        return params

    def __parse_response(self, response) -> AIResponse:
        message = response.choices[0].message
        tool_calls = None
        content = None

        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = [
                ToolCall(
                    request_call_id=response.id,
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )
                for tool_call in message.tool_calls
            ]
        else:
            content = message.content

        # Neither field is part of OpenAI's own Chat Completions schema
        # (OpenAI never returns reasoning text on this API, only a
        # `reasoning_tokens` count) — but OpenAI-compatible endpoints that
        # do return it use different non-standard field names: `reasoning`
        # (Ollama) vs `reasoning_content` (DeepSeek and others). The SDK's
        # message model allows extra fields, so this is a no-op (stays
        # None) against real OpenAI.
        thinking = getattr(message, "reasoning", None) or getattr(
            message, "reasoning_content", None
        )

        return AIResponse(
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
        )

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> AIResponse:
        """
        Generate messages.

        Args:
                messages (List[Dict[str, Any]]): _description_
                tools (Optional[List[Dict[str, Any]]], optional): _description_. Defaults to None.

        Raises:
                AIGooChatException: _description_

        Returns:
                AIResponse: _description_
        """
        try:
            params = self.__build_params(messages, tools, **kwargs)
            response = self.__call_openai(params)
            return self.__parse_response(response)

        except Exception as e:
            if isinstance(e, LLMfyException):
                raise  # Already handled, re-raise as-is
            raise LLMfyException(str(e), raw_error=e) from e

    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> AIResponse:
        """Async version of `generate` — uses `openai.AsyncOpenAI` natively
        (no thread offload). See `generate` for behavior/args."""
        try:
            params = self.__build_params(messages, tools, **kwargs)
            response = await self.__call_openai_async(params)
            return self.__parse_response(response)

        except Exception as e:
            if isinstance(e, LLMfyException):
                raise  # Already handled, re-raise as-is
            raise LLMfyException(str(e), raw_error=e) from e

    def __build_stream_params(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        **kwargs,
    ) -> dict[str, Any]:
        # Deliberately NOT the same shape as __build_params (generate's) —
        # frequency_penalty/presence_penalty/top_p/explicit stream=False are
        # not sent here; pre-existing divergence, preserved as-is rather than
        # unified, since changing request params is outside this task's scope.
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            **kwargs,
        }
        if self.config.max_tokens is not None:
            params["max_tokens"] = self.config.max_tokens
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature

        if self.config.thinking.enabled:
            params["reasoning_effort"] = self.config.thinking.effort or "medium"

        if tools:
            params["tools"] = [
                {"type": "function", "function": tool} for tool in tools
            ]
            params["tool_choice"] = "auto"

        return params

    def __process_stream_chunk(
        self, chunk, tool_calls_accumulator: dict[int, Any]
    ) -> AIResponse | None:
        """Process one Chat Completions stream chunk.

        Mutates `tool_calls_accumulator` in place (keyed by `tool_call.index`
        — see the comment below for why) and returns `None` for chunks with
        no `choices` (e.g. the trailing usage-only chunk from
        `stream_options={"include_usage": True}`) — shared by the sync
        (`generate_stream`) and async (`agenerate_stream`) streaming loops,
        which differ only in their iteration protocol (`for` vs `async for`).
        """
        if not chunk.choices:
            return None

        content = None
        thinking = None
        tool_calls = None

        delta = chunk.choices[0].delta

        if delta.content is not None:
            content = delta.content

        # See the matching comment in generate() — no-op against real
        # OpenAI, picks up reasoning deltas on compatible endpoints that
        # emit them (Ollama, DeepSeek, etc.).
        reasoning_delta = getattr(delta, "reasoning", None) or getattr(
            delta, "reasoning_content", None
        )
        if reasoning_delta is not None:
            thinking = reasoning_delta

        if delta.tool_calls is not None:
            tool_calls = []
            for tool_call in delta.tool_calls:
                # `index` is the position of this tool call among the
                # response's tool calls (0, 1, 2, ...) and is present on
                # EVERY chunk belonging to it — unlike `id`/`function.name`,
                # which only appear on that tool call's first chunk.
                # Correlating by `index` (rather than "whichever entry
                # happens to be first in the accumulator dict") is what
                # makes this correct when the model streams more than one
                # tool call in parallel: with 2+ tool calls in flight,
                # `next(iter(accumulator.values()))` would always grab the
                # first one, appending every tool call's argument deltas
                # onto it regardless of which one they actually belong to.
                index = tool_call.index

                if index not in tool_calls_accumulator:  # First chunk of this tool call
                    tool_calls_accumulator[index] = {
                        "request_call_id": chunk.id,
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": "",
                    }

                active_tool_call = tool_calls_accumulator[index]
                # Accumulate arguments across multiple chunks
                active_tool_call["arguments"] += tool_call.function.arguments or ""

                # Try to parse accumulated JSON when complete
                try:
                    parsed_arguments = json.loads(active_tool_call["arguments"])

                    # Construct the ToolCall object
                    tool_calls.append(
                        ToolCall(
                            request_call_id=active_tool_call["request_call_id"],
                            tool_call_id=active_tool_call["tool_call_id"],
                            name=active_tool_call["name"],
                            arguments=parsed_arguments,
                        )
                    )

                    # Remove the tool call once fully processed
                    del tool_calls_accumulator[index]

                except json.JSONDecodeError:
                    # JSON is incomplete, continue accumulating
                    pass

        return AIResponse(
            content=content,
            thinking=thinking,
            tool_calls=tool_calls if tool_calls else None,
        )

    def generate_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> Any:
        """
        Generate messages with streaming.

        Note:
                When using stream=True, the response does not include total usage information (usage field with prompt_tokens, completion_tokens, and total_tokens).

                Why?

                \t- In streaming mode, tokens are sent incrementally, so the API doesnt return a single final response that includes token usage.
                \t- If you need token usage, you must track tokens manually or make a separate non-streaming request.

        Args:
                messages (List[Dict[str, Any]]): _description_
                tools (Optional[List[Dict[str, Any]]], optional): _description_. Defaults to None.

        Raises:
                AIGooChatException: _description_

        Returns:
                Any: _description_
        """
        try:
            params = self.__build_stream_params(messages, tools, **kwargs)
            stream = self.__call_stream_openai(params)
            tool_calls_accumulator: dict[int, Any] = {}

            for chunk in stream:
                ai_response = self.__process_stream_chunk(chunk, tool_calls_accumulator)
                if ai_response is not None:
                    yield ai_response

        except Exception as e:
            if isinstance(e, LLMfyException):
                raise  # Already handled, re-raise as-is
            raise LLMfyException(str(e), raw_error=e) from e

    async def agenerate_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> AsyncGenerator[AIResponse, Any]:
        """Async version of `generate_stream` — uses `openai.AsyncOpenAI`
        natively (no thread offload). See `generate_stream` for behavior/args.
        """
        try:
            params = self.__build_stream_params(messages, tools, **kwargs)
            stream = self.__call_stream_openai_async(params)
            tool_calls_accumulator: dict[int, Any] = {}

            async for chunk in stream:
                ai_response = self.__process_stream_chunk(chunk, tool_calls_accumulator)
                if ai_response is not None:
                    yield ai_response

        except Exception as e:
            if isinstance(e, LLMfyException):
                raise  # Already handled, re-raise as-is
            raise LLMfyException(str(e), raw_error=e) from e
