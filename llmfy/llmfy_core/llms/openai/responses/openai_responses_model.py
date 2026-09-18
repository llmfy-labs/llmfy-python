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
from llmfy.llmfy_core.llms.openai.responses.openai_responses_config import (
    OpenAIResponsesConfig,
)
from llmfy.llmfy_core.messages.tool_call import ToolCall
from llmfy.llmfy_core.model_backend import ModelBackend
from llmfy.llmfy_core.responses.ai_response import AIResponse
from llmfy.llmfy_core.service_provider import ServiceProvider


class OpenAIResponsesModel(BaseAIModel):
    """
    OpenAIResponsesModel class — talks to OpenAI's Responses API
    (https://developers.openai.com/api/reference/responses/overview) instead
    of Chat Completions (see `OpenAIChatModel` for that variant).

    Example:
    ```python
    # Configuration
    config = OpenAIResponsesConfig(
            temperature=0.7
    )
    llm = OpenAIResponsesModel(model="gpt-5.6-terra", config=config)
    ...
    ```
    """

    def __init__(
        self,
        model: str,
        config: OpenAIResponsesConfig | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ):
        """
        OpenAIResponsesModel

        Args:
            model (str): Model ID
            config (OpenAIResponsesConfig, optional): Configuration. Defaults to OpenAIResponsesConfig().
            api_key (str, optional): OpenAI API key. Defaults to the `OPENAI_API_KEY`
                environment variable if not provided.
            base_url (str, optional): Base URL for the OpenAI API. Defaults to None,
                which uses the OpenAI SDK's default base URL.
            default_headers (dict[str, str], optional): Extra HTTP headers sent on every
                request, passed straight through to the `openai` SDK client. Needed for
                compatible endpoints that require headers beyond `Authorization` — e.g.
                Bedrock Mantle's Project-scoping headers.
        """
        config = config if config is not None else OpenAIResponsesConfig()
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

        self.client = openai.OpenAI(
            api_key=api_key, base_url=base_url, default_headers=default_headers
        )
        # Native async client for `agenerate` — see the matching comment in
        # `OpenAIChatModel.__init__`.
        self.async_client = openai.AsyncOpenAI(
            api_key=api_key, base_url=base_url, default_headers=default_headers
        )
        self.backend = ModelBackend.OPENAI_RESPONSES
        self.provider = ServiceProvider.OPENAI
        self.model_name = model
        self.config = config

    def __call_openai_responses(self, params: dict[str, Any]):
        # Import the decorator when the method is first defined/called
        import openai

        from llmfy.exception.exception_handler import handle_openai_error
        from llmfy.llmfy_core.llms.openai.responses.openai_responses_usage import (
            track_openai_responses_usage,
        )

        @track_openai_responses_usage
        def _call_openai_responses_impl(params: dict[str, Any]):
            try:
                response = self.client.responses.create(**params)
                return response
            except openai.APIError as e:
                raise handle_openai_error(e) from e
            # Any non-openai.APIError exceptions will naturally propagate up the call stack.

        return _call_openai_responses_impl(params)

    def __call_openai_responses_async(self, params: dict[str, Any]):
        # Async counterpart of `__call_openai_responses`, used by `agenerate`.
        import openai

        from llmfy.exception.exception_handler import handle_openai_error
        from llmfy.llmfy_core.llms.openai.responses.openai_responses_usage import (
            track_openai_responses_usage,
        )

        @track_openai_responses_usage
        async def _call_openai_responses_impl_async(params: dict[str, Any]):
            try:
                response = await self.async_client.responses.create(**params)
                return response
            except openai.APIError as e:
                raise handle_openai_error(e) from e
            # Any non-openai.APIError exceptions will naturally propagate up the call stack.

        return _call_openai_responses_impl_async(params)

    def __call_stream_openai_responses(self, params: dict[str, Any]):
        # Import the decorator when the method is first defined/called
        import openai

        from llmfy.exception.exception_handler import handle_openai_error
        from llmfy.llmfy_core.llms.openai.responses.openai_responses_usage import (
            track_openai_responses_stream_usage,
        )

        @track_openai_responses_stream_usage
        def __call_stream_openai_responses_impl(params: dict[str, Any]):
            try:
                params["stream"] = True
                return self.client.responses.create(**params)
            except openai.APIError as e:
                raise handle_openai_error(e) from e
            # Any non-openai.APIError exceptions will naturally propagate up the call stack.

        return __call_stream_openai_responses_impl(params)

    def __call_stream_openai_responses_async(self, params: dict[str, Any]):
        # Async counterpart of `__call_stream_openai_responses`, used by
        # `agenerate_stream`.
        import openai

        from llmfy.exception.exception_handler import handle_openai_error
        from llmfy.llmfy_core.llms.openai.responses.openai_responses_usage import (
            track_openai_responses_stream_usage_async,
        )

        @track_openai_responses_stream_usage_async
        async def __call_stream_openai_responses_impl_async(params: dict[str, Any]):
            try:
                params["stream"] = True
                stream = await self.async_client.responses.create(**params)
                async for event in stream:
                    yield event
            except openai.APIError as e:
                raise handle_openai_error(e) from e

        return __call_stream_openai_responses_impl_async(params)

    def __to_responses_input(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Flattens `MessageBufferBuilder.get_messages()`'s output into the Responses API's
        flat `input` item array — unwrapping `OpenAIResponsesFormatter`'s
        private `{"__items__": [...]}` convention for messages that expand to
        more than one item (e.g. an assistant turn with several tool calls)."""
        input_items: list[dict[str, Any]] = []
        for message in messages:
            items = message.get("__items__")
            if items is not None:
                input_items.extend(items)
            else:
                input_items.append(message)
        return input_items

    def __build_params(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        **kwargs,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model_name,
            "input": self.__to_responses_input(messages),
            "max_output_tokens": self.config.max_output_tokens,
            **kwargs,
        }
        # Omitted entirely (not sent as null) when None — some models reject
        # these params outright rather than accepting a default for them.
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            params["top_p"] = self.config.top_p

        if self.config.reasoning.enabled:
            reasoning: dict[str, Any] = {
                "effort": self.config.reasoning.effort or "medium"
            }
            if self.config.reasoning.summary:
                reasoning["summary"] = self.config.reasoning.summary
            params["reasoning"] = reasoning

        if tools:
            params["tools"] = [{"type": "function", **tool} for tool in tools]
            params["tool_choice"] = "auto"

        return params

    def __parse_response(self, response) -> AIResponse:
        content = None
        thinking = None
        tool_calls = None

        for item in response.output:
            if item.type == "function_call":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    ToolCall(
                        request_call_id=response.id,
                        tool_call_id=item.call_id,
                        name=item.name,
                        arguments=json.loads(item.arguments),
                    )
                )
            elif item.type == "message" and content is None:
                content = "".join(
                    c.text
                    for c in item.content
                    if getattr(c, "type", None) == "output_text"
                )
            elif item.type == "reasoning" and item.summary:
                # Populated only when `reasoning.summary` is requested in
                # the config — the full chain-of-thought itself is never
                # returned, only this model-generated summary.
                thinking = "".join(s.text for s in item.summary)

        # A turn that requests tool calls takes priority — mirrors OpenAIChatModel's
        # Chat Completions behavior of not surfacing partial text alongside tool_calls.
        if tool_calls:
            content = None

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
        Generate messages via the Responses API.

        Args:
                messages (List[Dict[str, Any]]): _description_
                tools (Optional[List[Dict[str, Any]]], optional): _description_. Defaults to None.

        Raises:
                LLMfyException: _description_

        Returns:
                AIResponse: _description_
        """
        try:
            params = self.__build_params(messages, tools, **kwargs)
            params["stream"] = False

            response = self.__call_openai_responses(params)
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
            params["stream"] = False

            response = await self.__call_openai_responses_async(params)
            return self.__parse_response(response)

        except Exception as e:
            if isinstance(e, LLMfyException):
                raise  # Already handled, re-raise as-is
            raise LLMfyException(str(e), raw_error=e) from e

    def __process_stream_event(
        self,
        event,
        pending_tool_calls: dict[str, dict[str, Any]],
        response_id: str | None,
    ) -> tuple[AIResponse | None, str | None]:
        """Process one Responses-API stream event.

        Mutates `pending_tool_calls` in place and returns `(ai_response,
        response_id)` — the caller's loop carries `response_id` into the
        next call since it's set once (on `response.created`) and read later
        (on `response.function_call_arguments.done`). Shared by the sync
        (`generate_stream`) and async (`agenerate_stream`) streaming loops,
        which differ only in their iteration protocol (`for` vs `async for`).
        Most event types produce no `AIResponse` (returns `None` for those —
        the caller only yields non-`None` results).
        """
        event_type = getattr(event, "type", None)

        if event_type == "response.created":
            response_id = event.response.id

        elif event_type == "response.output_item.added":
            item = event.item
            if getattr(item, "type", None) == "function_call":
                pending_tool_calls[item.id] = {
                    "call_id": item.call_id,
                    "name": item.name,
                }

        elif event_type == "response.output_text.delta":
            if event.delta:
                return AIResponse(content=event.delta), response_id

        elif event_type == "response.reasoning_summary_text.delta":
            if event.delta:
                return AIResponse(thinking=event.delta), response_id

        elif event_type == "response.function_call_arguments.done":
            pending = pending_tool_calls.pop(event.item_id, None)
            if pending:
                return (
                    AIResponse(
                        tool_calls=[
                            ToolCall(
                                request_call_id=response_id or "",
                                tool_call_id=pending["call_id"],
                                name=pending["name"],
                                arguments=json.loads(event.arguments),
                            )
                        ]
                    ),
                    response_id,
                )

        elif event_type == "error":
            raise LLMfyException(
                getattr(event, "message", "OpenAI Responses stream error"),
                raw_error=event,
            )

        return None, response_id

    def generate_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> Any:
        """
        Generate messages via the Responses API with streaming.

        Args:
                messages (List[Dict[str, Any]]): _description_
                tools (Optional[List[Dict[str, Any]]], optional): _description_. Defaults to None.

        Raises:
                LLMfyException: _description_

        Returns:
                Any: _description_
        """
        try:
            params = self.__build_params(messages, tools, **kwargs)
            stream = self.__call_stream_openai_responses(params)

            response_id = None
            # Tracks in-flight function_call items by their output item id, keyed
            # at `response.output_item.added` and finalized at
            # `response.function_call_arguments.done`.
            pending_tool_calls: dict[str, dict[str, Any]] = {}

            for event in stream:
                ai_response, response_id = self.__process_stream_event(
                    event, pending_tool_calls, response_id
                )
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
            params = self.__build_params(messages, tools, **kwargs)
            stream = self.__call_stream_openai_responses_async(params)

            response_id = None
            pending_tool_calls: dict[str, dict[str, Any]] = {}

            async for event in stream:
                ai_response, response_id = self.__process_stream_event(
                    event, pending_tool_calls, response_id
                )
                if ai_response is not None:
                    yield ai_response

        except Exception as e:
            if isinstance(e, LLMfyException):
                raise  # Already handled, re-raise as-is
            raise LLMfyException(str(e), raw_error=e) from e
