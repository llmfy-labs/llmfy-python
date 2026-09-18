try:
    from google import genai
except ImportError:
    genai = None

import uuid
from collections.abc import AsyncGenerator
from typing import Any

from llmfy.exception.llmfy_exception import LLMfyException
from llmfy.llmfy_core.llms.base_ai_model import BaseAIModel
from llmfy.llmfy_core.llms.google.generate.googleai_generate_config import (
    GoogleAIGenerateConfig,
)
from llmfy.llmfy_core.messages.tool_call import ToolCall
from llmfy.llmfy_core.model_backend import ModelBackend
from llmfy.llmfy_core.responses.ai_response import AIResponse
from llmfy.llmfy_core.service_provider import ServiceProvider


class GoogleAIGenerateModel(BaseAIModel):
    """
    GoogleAIGenerateModel class.

    Uses the `google-genai` package to interact with Google AI (Gemini) models.

    Example:
    ```python
    # Configuration
    config = GoogleAIGenerateConfig(
            temperature=0.7
    )
    llm = GoogleAIGenerateModel(model="gemini-2.0-flash", config=config)
    ...
    ```
    """

    def __init__(
        self,
        model: str,
        config: GoogleAIGenerateConfig | None = None,
        api_key: str | None = None,
    ):
        """
        GoogleAIGenerateModel

        Args:
            model (str): Model ID (e.g. "gemini-2.0-flash")
            config (GoogleAIGenerateConfig, optional): Configuration. Defaults to GoogleAIGenerateConfig().
            api_key (str, optional): Google AI API key. Defaults to the `GOOGLE_API_KEY`
                environment variable if not provided.
        """
        config = config if config is not None else GoogleAIGenerateConfig()
        if genai is None:
            raise LLMfyException(
                'google-genai package is not installed. Install it using `pip install "llmfy[google-genai]"`'
            )

        import os

        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise LLMfyException(
                "Please provide `GOOGLE_API_KEY` on your environment or pass `api_key`!"
            )

        self.client = genai.Client(api_key=api_key)
        self.backend = ModelBackend.GOOGLE_GENERATE
        self.provider = ServiceProvider.GOOGLE
        self.model_name = model
        self.config = config

    def __build_config(self, tools=None, system_instruction=None):
        """Build GenerateContentConfig from self.config."""
        from google.genai import types

        config_kwargs: dict[str, Any] = {
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            config_kwargs["max_output_tokens"] = self.config.max_tokens
        if self.config.top_p is not None:
            config_kwargs["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            config_kwargs["top_k"] = self.config.top_k
        if self.config.stop_sequences is not None:
            config_kwargs["stop_sequences"] = self.config.stop_sequences
        if self.config.candidate_count is not None:
            config_kwargs["candidate_count"] = self.config.candidate_count
        if self.config.seed is not None:
            config_kwargs["seed"] = self.config.seed
        if self.config.presence_penalty is not None:
            config_kwargs["presence_penalty"] = self.config.presence_penalty
        if self.config.frequency_penalty is not None:
            config_kwargs["frequency_penalty"] = self.config.frequency_penalty
        if self.config.response_mime_type is not None:
            config_kwargs["response_mime_type"] = self.config.response_mime_type
        if self.config.response_schema is not None:
            config_kwargs["response_schema"] = self.config.response_schema
        if self.config.safety_settings is not None:
            config_kwargs["safety_settings"] = self.config.safety_settings
        # Pass pre-created cache reference when provided (explicit caching)
        if self.config.prompt_caching.cached_content is not None:
            config_kwargs["cached_content"] = self.config.prompt_caching.cached_content
        if self.config.thinking.raw is not None:
            # Raw override takes priority (backward compat)
            config_kwargs["thinking_config"] = self.config.thinking.raw
        elif self.config.thinking.enabled:
            thinking_kwargs: dict[str, Any] = {}
            if self.config.thinking.budget_tokens is not None:
                thinking_kwargs["thinking_budget"] = self.config.thinking.budget_tokens
            if self.config.thinking.level is not None:
                thinking_kwargs["thinking_level"] = self.config.thinking.level
            if self.config.thinking.include_thoughts is not None:
                thinking_kwargs["include_thoughts"] = (
                    self.config.thinking.include_thoughts
                )
            config_kwargs["thinking_config"] = types.ThinkingConfig(**thinking_kwargs)
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = [types.Tool(function_declarations=tools)]

        return types.GenerateContentConfig(**config_kwargs)

    def __call_googleai(self, params: dict[str, Any]):
        import httpx
        from google.genai import errors

        from llmfy.exception.exception_handler import handle_google_error
        from llmfy.llmfy_core.llms.google.generate.googleai_generate_usage import (
            track_googleai_usage,
        )

        @track_googleai_usage
        def _call_googleai_impl(params: dict[str, Any]):
            try:
                response = self.client.models.generate_content(
                    model=params["model"],
                    contents=params["contents"],
                    config=params["config"],
                )
                return response
            except (errors.APIError, httpx.TimeoutException) as e:
                raise handle_google_error(e) from e

        return _call_googleai_impl(params)

    def __call_googleai_async(self, params: dict[str, Any]):
        # Async counterpart of `__call_googleai`, used by `agenerate`. The
        # `google-genai` SDK has no separate async client class — the same
        # `genai.Client` exposes an async surface under `.aio`.
        import httpx
        from google.genai import errors

        from llmfy.exception.exception_handler import handle_google_error
        from llmfy.llmfy_core.llms.google.generate.googleai_generate_usage import (
            track_googleai_usage,
        )

        @track_googleai_usage
        async def _call_googleai_impl_async(params: dict[str, Any]):
            try:
                response = await self.client.aio.models.generate_content(
                    model=params["model"],
                    contents=params["contents"],
                    config=params["config"],
                )
                return response
            except (errors.APIError, httpx.TimeoutException) as e:
                raise handle_google_error(e) from e

        return _call_googleai_impl_async(params)

    def __call_stream_googleai(self, params: dict[str, Any]):
        import httpx
        from google.genai import errors

        from llmfy.exception.exception_handler import handle_google_error
        from llmfy.llmfy_core.llms.google.generate.googleai_generate_usage import (
            track_googleai_stream_usage,
        )

        @track_googleai_stream_usage
        def _call_stream_googleai_impl(params: dict[str, Any]):
            try:
                return self.client.models.generate_content_stream(
                    model=params["model"],
                    contents=params["contents"],
                    config=params["config"],
                )
            except (errors.APIError, httpx.TimeoutException) as e:
                raise handle_google_error(e) from e

        return _call_stream_googleai_impl(params)

    def __call_stream_googleai_async(self, params: dict[str, Any]):
        # Async counterpart of `__call_stream_googleai`, used by
        # `agenerate_stream`. Same `.aio` async surface as `__call_googleai_async`.
        import httpx
        from google.genai import errors

        from llmfy.exception.exception_handler import handle_google_error
        from llmfy.llmfy_core.llms.google.generate.googleai_generate_usage import (
            track_googleai_stream_usage_async,
        )

        @track_googleai_stream_usage_async
        async def _call_stream_googleai_impl_async(params: dict[str, Any]):
            try:
                stream = await self.client.aio.models.generate_content_stream(
                    model=params["model"],
                    contents=params["contents"],
                    config=params["config"],
                )
                async for chunk in stream:
                    yield chunk
            except (errors.APIError, httpx.TimeoutException) as e:
                raise handle_google_error(e) from e

        return _call_stream_googleai_impl_async(params)

    def __build_generate_params(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        system_instruction = next(
            (
                msg["parts"][0]["text"]
                for msg in messages
                if msg.get("role") == "system"
            ),
            None,
        )
        contents = [msg for msg in messages if msg.get("role") != "system"]

        return {
            "model": self.model_name,
            "contents": contents,
            "config": self.__build_config(
                tools=tools, system_instruction=system_instruction
            ),
        }

    def __parse_response(self, response) -> AIResponse:
        request_call_id = str(uuid.uuid4())
        tool_calls = None
        content = None
        thinking = None

        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                function_call_parts = [
                    p
                    for p in candidate.content.parts
                    if p.function_call is not None
                ]
                if function_call_parts:
                    tool_calls = [
                        ToolCall(
                            request_call_id=request_call_id,
                            tool_call_id=fc.id if fc.id else str(uuid.uuid4()),
                            name=fc.name or "",
                            arguments=dict(fc.args) if fc.args else {},
                        )
                        for part in function_call_parts
                        for fc in [part.function_call]
                        if fc is not None
                    ]
                else:
                    # response.text already excludes thought=True parts
                    content = response.text

                thinking_parts = [
                    p.text
                    for p in candidate.content.parts
                    if getattr(p, "thought", False) and p.text
                ]
                if thinking_parts:
                    thinking = "".join(thinking_parts)

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
            messages (List[Dict[str, Any]]): Formatted messages from MessageBufferBuilder.get_messages().
            tools (Optional[List[Dict[str, Any]]], optional): Tool function definitions. Defaults to None.

        Returns:
            AIResponse: Response with content or tool_calls.
        """
        try:
            params = self.__build_generate_params(messages, tools)
            response = self.__call_googleai(params)
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
        """Async version of `generate` — uses the `google-genai` SDK's async
        surface (`client.aio.models.generate_content`) natively, no thread
        offload. See `generate` for behavior/args."""
        try:
            params = self.__build_generate_params(messages, tools)
            response = await self.__call_googleai_async(params)
            return self.__parse_response(response)

        except Exception as e:
            if isinstance(e, LLMfyException):
                raise  # Already handled, re-raise as-is
            raise LLMfyException(str(e), raw_error=e) from e

    def __process_stream_chunk(self, chunk, request_call_id: str) -> AIResponse:
        """Process one generate_content_stream chunk. Google delivers complete
        function_call objects in a single chunk (no incremental argument
        accumulation needed, unlike OpenAI), so unlike the other backends'
        per-chunk helpers, no mutable accumulator needs to be threaded
        between calls — `request_call_id` is fixed for the whole stream.
        Shared by the sync (`generate_stream`) and async (`agenerate_stream`)
        streaming loops, which differ only in their iteration protocol
        (`for` vs `async for`).
        """
        content = None
        thinking = None
        tool_calls = None

        if chunk.candidates:
            for candidate in chunk.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.text is not None:
                            if getattr(part, "thought", False):
                                thinking = part.text
                            else:
                                content = part.text

                        if part.function_call is not None:
                            fc = part.function_call
                            fc_id = fc.id if fc.id else str(uuid.uuid4())
                            # Google delivers complete function calls in one chunk
                            tool_calls = [
                                ToolCall(
                                    request_call_id=request_call_id,
                                    tool_call_id=fc_id,
                                    name=fc.name or "",
                                    arguments=dict(fc.args) if fc.args else {},
                                )
                            ]

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
            Google AI delivers complete function_call objects in a single chunk
            (no incremental argument accumulation needed, unlike OpenAI).

        Args:
            messages (List[Dict[str, Any]]): Formatted messages from MessageBufferBuilder.get_messages().
            tools (Optional[List[Dict[str, Any]]], optional): Tool function definitions. Defaults to None.

        Returns:
            Generator[AIResponse]: Yields AIResponse chunks.
        """
        try:
            params = self.__build_generate_params(messages, tools)
            stream = self.__call_stream_googleai(params)
            request_call_id = str(uuid.uuid4())

            for chunk in stream:
                yield self.__process_stream_chunk(chunk, request_call_id)

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
        """Async version of `generate_stream` — uses the `google-genai` SDK's
        async surface (`client.aio.models.generate_content_stream`) natively,
        no thread offload. See `generate_stream` for behavior/args."""
        try:
            params = self.__build_generate_params(messages, tools)
            stream = self.__call_stream_googleai_async(params)
            request_call_id = str(uuid.uuid4())

            async for chunk in stream:
                yield self.__process_stream_chunk(chunk, request_call_id)

        except Exception as e:
            if isinstance(e, LLMfyException):
                raise  # Already handled, re-raise as-is
            raise LLMfyException(str(e), raw_error=e) from e
