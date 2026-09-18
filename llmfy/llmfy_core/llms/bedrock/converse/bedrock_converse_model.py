try:
    import boto3
except ImportError:
    boto3 = None

try:
    import aioboto3
except ImportError:
    aioboto3 = None

import json
import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any, cast

from llmfy.exception.llmfy_exception import LLMfyException
from llmfy.llmfy_core.llms.base_ai_model import BaseAIModel
from llmfy.llmfy_core.llms.bedrock.converse.bedrock_converse_config import (
    BedrockConverseConfig,
)
from llmfy.llmfy_core.messages.tool_call import ToolCall
from llmfy.llmfy_core.model_backend import ModelBackend
from llmfy.llmfy_core.responses.ai_response import AIResponse
from llmfy.llmfy_core.service_provider import ServiceProvider


class BedrockConverseModel(BaseAIModel):
    """
    BedrockConverseModel class.

    Example:
    ```python
    # Configuration
    config = BedrockConverseConfig(
            temperature=0.7
    )
    llm = BedrockConverseModel(model="amazon.nova-pro-v1:0", config=config)
    ...
    ```
    """

    def __init__(
        self,
        model: str,
        config: BedrockConverseConfig | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_bedrock_region: str | None = None,
    ):
        """
        BedrockConverseModel

        Args:
            model (str): Model ID
            config (BedrockConverseConfig, optional): Configuration. Defaults to BedrockConverseConfig().
            aws_access_key_id (str, optional): AWS access key ID. Defaults to the
                `AWS_ACCESS_KEY_ID` environment variable if not provided.
            aws_secret_access_key (str, optional): AWS secret access key. Defaults to
                the `AWS_SECRET_ACCESS_KEY` environment variable if not provided.
            aws_bedrock_region (str, optional): AWS Bedrock region. Defaults to the
                `AWS_BEDROCK_REGION` environment variable if not provided.
        """
        config = config if config is not None else BedrockConverseConfig()
        if boto3 is None:
            raise LLMfyException(
                'boto3 package is not installed. Install it using `pip install "llmfy[boto3]"`'
            )

        aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = aws_secret_access_key or os.getenv(
            "AWS_SECRET_ACCESS_KEY"
        )
        aws_bedrock_region = aws_bedrock_region or os.getenv("AWS_BEDROCK_REGION")

        if not aws_access_key_id:
            raise LLMfyException(
                "Please provide `AWS_ACCESS_KEY_ID` on your environment or pass `aws_access_key_id`!"
            )
        if not aws_secret_access_key:
            raise LLMfyException(
                "Please provide `AWS_SECRET_ACCESS_KEY` on your environment or pass `aws_secret_access_key`!"
            )
        if not aws_bedrock_region:
            raise LLMfyException(
                "Please provide `AWS_BEDROCK_REGION` on your environment or pass `aws_bedrock_region`!"
            )

        self.backend = ModelBackend.BEDROCK_CONVERSE
        self.provider = ServiceProvider.BEDROCK
        self.model_name = model
        self.config = config
        self.client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_bedrock_region,
        )
        # Native async support for `agenerate`/`agenerate_stream` needs the
        # optional `aioboto3` dependency — boto3 itself has no async mode.
        # Built eagerly here (cheap: a Session, no connection opened until
        # the first request) so it's ready if/when the async methods are
        # called; stays None when aioboto3 isn't installed, and
        # `__require_aioboto3` raises a clear, actionable error at *that*
        # point instead — sync-only users should never need aioboto3
        # installed at all.
        self._aioboto3_session = (
            aioboto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_bedrock_region,
            )
            if aioboto3 is not None
            else None
        )

    def __require_aioboto3(self) -> None:
        if aioboto3 is None:
            raise LLMfyException(
                "aioboto3 package is not installed. It is required for async "
                "(agenerate/agenerate_stream) calls on Bedrock — boto3 itself has "
                'no async mode. Install it using `pip install "llmfy[aioboto3]"`.'
            )

    def __call_bedrock(self, params: dict[str, Any]):
        # Import the decorator when the method is first defined/called
        from botocore.exceptions import (
            ClientError,
            ConnectTimeoutError,
            ReadTimeoutError,
        )

        from llmfy.exception.exception_handler import handle_bedrock_error
        from llmfy.llmfy_core.llms.bedrock.converse.bedrock_converse_usage import (
            track_bedrock_converse_usage,
        )

        @track_bedrock_converse_usage
        def _call_bedrock_impl(params: dict[str, Any]):
            try:
                response = self.client.converse(**params)
                return response
            except (ClientError, ReadTimeoutError, ConnectTimeoutError) as e:
                raise handle_bedrock_error(e) from e

        return _call_bedrock_impl(params)

    def __call_stream_bedrock(self, params: dict[str, Any]):
        # Import the decorator when the method is first defined/called
        from botocore.exceptions import (
            ClientError,
            ConnectTimeoutError,
            ReadTimeoutError,
        )

        from llmfy.exception.exception_handler import handle_bedrock_error
        from llmfy.llmfy_core.llms.bedrock.converse.bedrock_converse_usage import (
            track_bedrock_converse_stream_usage,
        )

        @track_bedrock_converse_stream_usage
        def _call_stream_bedrock_impl(params: dict[str, Any]):
            try:
                return self.client.converse_stream(**params)
            except (ClientError, ReadTimeoutError, ConnectTimeoutError) as e:
                raise handle_bedrock_error(e) from e

        return _call_stream_bedrock_impl(params)

    def __call_bedrock_async(self, params: dict[str, Any]):
        # Async counterpart of `__call_bedrock`, used by `agenerate`.
        from botocore.exceptions import (
            ClientError,
            ConnectTimeoutError,
            ReadTimeoutError,
        )

        from llmfy.exception.exception_handler import handle_bedrock_error
        from llmfy.llmfy_core.llms.bedrock.converse.bedrock_converse_usage import (
            track_bedrock_converse_usage,
        )

        # `agenerate` already calls `__require_aioboto3()` before reaching
        # here, so this is always non-None at this point — asserted (rather
        # than left implicit) purely to narrow the type for static checkers.
        assert self._aioboto3_session is not None
        session = self._aioboto3_session

        @track_bedrock_converse_usage
        async def _call_bedrock_impl_async(params: dict[str, Any]):
            try:
                async with cast(Any, session.client("bedrock-runtime")) as client:
                    return await client.converse(**params)
            except (ClientError, ReadTimeoutError, ConnectTimeoutError) as e:
                raise handle_bedrock_error(e) from e

        return _call_bedrock_impl_async(params)

    def __call_bedrock_stream_async(self, params: dict[str, Any]):
        # Async counterpart of `__call_stream_bedrock`, used by
        # `agenerate_stream`. Yields raw Converse-stream events directly
        # (unlike the sync path, which returns a response dict with a
        # "stream" key for the caller to iterate later) because the
        # aioboto3 client's underlying connection closes when the
        # `async with` block below exits — the stream must be fully
        # consumed from inside it.
        from botocore.exceptions import (
            ClientError,
            ConnectTimeoutError,
            ReadTimeoutError,
        )

        from llmfy.exception.exception_handler import handle_bedrock_error
        from llmfy.llmfy_core.llms.bedrock.converse.bedrock_converse_usage import (
            track_bedrock_converse_stream_usage_async,
        )

        # `agenerate_stream` already calls `__require_aioboto3()` before
        # reaching here, so this is always non-None at this point — asserted
        # (rather than left implicit) purely to narrow the type for static
        # checkers.
        assert self._aioboto3_session is not None
        session = self._aioboto3_session

        @track_bedrock_converse_stream_usage_async
        async def _call_stream_bedrock_impl_async(params: dict[str, Any]):
            try:
                async with cast(Any, session.client("bedrock-runtime")) as client:
                    response = await client.converse_stream(**params)
                    async for event in response["stream"]:
                        yield event
            except (ClientError, ReadTimeoutError, ConnectTimeoutError) as e:
                raise handle_bedrock_error(e) from e

        return _call_stream_bedrock_impl_async(params)

    def __build_params(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        **kwargs,
    ) -> dict[str, Any]:
        _system = next(
            (msg["content"] for msg in messages if msg["role"] == "system"), None
        )
        _messages = [msg for msg in messages if msg["role"] != "system"]

        inferences = {
            "temperature": self.config.temperature,
            "maxTokens": self.config.max_tokens,
            "stopSequences": self.config.stopSequences,
            "topP": self.config.top_p,
        }
        # Remove None values
        inference_config = {
            key: value for key, value in inferences.items() if value is not None
        }

        additionals: dict[str, Any] = {
            "top_k": self.config.top_k,
            **kwargs,
        }

        if self.config.thinking.enabled:
            if self.config.thinking.reasoning_effort is not None:
                # Amazon Nova 2 Lite format
                additionals["reasoningConfig"] = {
                    "type": "enabled",
                    "maxReasoningEffort": self.config.thinking.reasoning_effort,
                }
            elif self.config.thinking.type == "adaptive":
                # Claude adaptive thinking (Sonnet/Opus 4.6, Fable 5, Mythos 5, Opus 4.7)
                additionals["thinking"] = {"type": "adaptive"}
                if self.config.thinking.effort is not None:
                    additionals["output_config"] = {
                        "effort": self.config.thinking.effort
                    }
            else:
                # Claude extended thinking (3.7 Sonnet, Claude 4 Sonnet/Opus/Haiku, 4.5 series)
                _thinking: dict[str, Any] = {"type": "enabled"}
                if self.config.thinking.budget_tokens is not None:
                    _thinking["budget_tokens"] = self.config.thinking.budget_tokens
                additionals["thinking"] = _thinking

        # Remove None values
        additional_config = {
            key: value for key, value in additionals.items() if value is not None
        }

        params = {
            "modelId": self.model_name,
            "messages": _messages,
            "inferenceConfig": inference_config,
            "additionalModelRequestFields": additional_config,
            **({"system": _system} if _system is not None else {}),
        }

        # Prompt caching: inject cachePoint markers for supported Claude models.
        # cachePoints are evaluated cumulatively: tools → system → messages.
        # A cachePoint after the system caches the system prompt (~90% savings).
        # A cachePoint at the end of the last message caches the full conversation
        # prefix so the next turn can serve it from cache.
        if self.config.prompt_caching.enabled:
            _cache_point: dict[str, Any] = {"type": "default"}
            if self.config.prompt_caching.ttl is not None:
                _cache_point["ttl"] = self.config.prompt_caching.ttl
            _cache_point_entry = {"cachePoint": _cache_point}

            if _system is not None:
                # Bedrock cache is a byte-identical prefix match,
                # if system prompt changes, which invalidates the cache every time
                # "You are an expert in Python"     → cache WRITE A (cached for 5 min)
                # "You are an expert in JavaScript" → cache WRITE B (different prefix, separate cache)
                # "You are an expert in Go"         → cache WRITE C (different prefix, separate cache)
                # "You are an expert in Python"     → cache READ D (hit! same as Call 1, still within 5 min)
                params["system"] = list(_system) + [_cache_point_entry]
            if _messages:
                # Bedrock writes only the DELTA (new tokens since the last cachePoint),
                # not the full prefix. Read pool grows each turn; write cost stays constant.
                #
                # Turn 1:  [user_q1 + cachePoint]
                #        → WRITE delta: system + user_q1          (all new, no prior cache)
                #
                # Turn 2:  [user_q1, assistant_r1, user_q2 + cachePoint]
                #        → READ  system + user_q1                 ← from Turn 1's cachePoint
                #        → WRITE delta: assistant_r1 + user_q2    ← only new tokens since Turn 1
                #
                # Turn 3:  [user_q1, assistant_r1, user_q2, assistant_r2, user_q3 + cachePoint]
                #        → READ  system + user_q1 + assistant_r1 + user_q2  ← from Turn 2's cachePoint
                #        → WRITE delta: assistant_r2 + user_q3    ← only new tokens since Turn 2
                #
                # Write cost is CONSTANT from Turn 2 onwards (delta per turn only).
                # Read pool GROWS each turn — savings increase with conversation length.
                _messages = list(_messages)
                last_msg = _messages[-1]
                last_content = list(last_msg.get("content", []))
                last_content.append(_cache_point_entry)
                _messages[-1] = {**last_msg, "content": last_content}
                params["messages"] = _messages

        if tools:
            """
            ToolConfig
            {
                "tools": [
                    {
                        "toolSpec": {
                            "name": "top_song",
                            "description": "Get the most popular song played on a radio station.",
                            "inputSchema": {
                                "json": {
                                    "type": "object",
                                    "properties": {
                                        "sign": {
                                            "type": "string",
                                            "description": "The call sign for the radio station for which you want the most popular song. Example calls signs are WZPZ and WKRP."
                                        }
                                    },
                                    "required": [
                                        "sign"
                                    ]
                                }
                            }
                        }
                    }
                ]
            }
            """
            params["toolConfig"] = {"tools": [{"toolSpec": tool} for tool in tools]}

        return params

    def __parse_response(self, response) -> AIResponse:
        output_message = response["output"]["message"]
        stop_reason = response["stopReason"]
        tool_calls = None
        content = None
        thinking = None

        if stop_reason == "tool_use":
            tool_requests = response["output"]["message"]["content"]
            tool_callings = []
            for tool_request in tool_requests:
                if "toolUse" in tool_request:
                    tool = tool_request["toolUse"]
                    tool_callings.append(
                        ToolCall(
                            request_call_id=response["ResponseMetadata"]["RequestId"],
                            tool_call_id=tool["toolUseId"],
                            name=tool["name"],
                            arguments=tool["input"],
                        )
                    )
            tool_calls = tool_callings
        else:
            # NOT output_message["content"][0]["text"] — when thinking is
            # enabled, the reasoningContent block comes first in the array,
            # so blindly indexing [0] grabs the reasoning block (no "text"
            # key) instead of the actual answer.
            for block in output_message["content"]:
                if "reasoningContent" in block:
                    thinking = block["reasoningContent"]["reasoningText"]["text"]
                elif "text" in block:
                    content = block["text"]

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
            response = self.__call_bedrock(params)
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
        """Async version of `generate` — uses `aioboto3` natively (no thread
        offload). Requires the optional `aioboto3` dependency (boto3 itself
        has no async mode); raises `LLMfyException` if it isn't installed.
        See `generate` for behavior/args."""
        try:
            self.__require_aioboto3()
            params = self.__build_params(messages, tools, **kwargs)
            response = await self.__call_bedrock_async(params)
            return self.__parse_response(response)
        except Exception as e:
            if isinstance(e, LLMfyException):
                raise  # Already handled, re-raise as-is
            raise LLMfyException(str(e), raw_error=e) from e

    def __process_stream_chunk(
        self,
        chunk: dict[str, Any],
        tools: list[dict[str, Any]],
        tool_use: dict[str, Any],
        request_id: str,
    ) -> tuple[AIResponse, dict[str, Any]]:
        """Process one Converse-stream event.

        Mutates `tools` (accumulated completed tool calls) in place and
        returns the (possibly new) `tool_use` dict the caller's loop should
        carry into the next chunk — shared by the sync (`generate_stream`)
        and async (`agenerate_stream`) streaming loops, which differ only in
        their iteration protocol (`for` vs `async for`).
        """
        text = None
        thinking = None

        if "contentBlockStart" in chunk:
            tool = chunk["contentBlockStart"]["start"]["toolUse"]
            tool_use["toolUseId"] = tool["toolUseId"]
            tool_use["name"] = tool["name"]

        if "contentBlockDelta" in chunk:
            delta = chunk["contentBlockDelta"]["delta"]
            if "text" in delta:
                text = delta["text"]

            if "reasoningContent" in delta:
                # {"text": "..."} while the reasoning is being generated,
                # then a final {"signature": "..."} delta with no text —
                # ignore the latter.
                if "text" in delta["reasoningContent"]:
                    thinking = delta["reasoningContent"]["text"]

            if "toolUse" in delta:
                if "input" not in tool_use:
                    tool_use["input"] = ""
                tool_use["input"] += delta["toolUse"]["input"]

        elif "contentBlockStop" in chunk:
            if "input" in tool_use:
                tool_use["input"] = json.loads(tool_use["input"])
                tools.append({"toolUse": tool_use})
                tool_use = {}

        tool_calls = []
        for tools_content in tools:
            if "toolUse" in tools_content:
                tool = tools_content["toolUse"]
                tool_calls.append(
                    ToolCall(
                        request_call_id=request_id,
                        tool_call_id=tool["toolUseId"],
                        name=tool["name"],
                        arguments=tool["input"],
                    )
                )

        response = AIResponse(
            content=text,
            thinking=thinking,
            tool_calls=tool_calls if tool_calls else None,
        )
        return response, tool_use

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
            params = self.__build_params(messages, tools, **kwargs)
            response = self.__call_stream_bedrock(params)
            res_metadata = response.get("ResponseMetadata")
            stream = response.get("stream")

            request_id: str = str(res_metadata.get("RequestId") or uuid.uuid4())
            tools_accum: list[dict[str, Any]] = []
            tool_use: dict[str, Any] = {}

            if stream:
                for chunk in stream:
                    ai_response, tool_use = self.__process_stream_chunk(
                        chunk, tools_accum, tool_use, request_id
                    )
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
        """Async version of `generate_stream` — uses `aioboto3` natively (no
        thread offload). Requires the optional `aioboto3` dependency; raises
        `LLMfyException` if it isn't installed. See `generate_stream` for
        behavior/args.

        Note: `request_call_id` on yielded `ToolCall`s is a synthesized uuid
        rather than the real Bedrock RequestId (unlike `generate_stream`) —
        aioboto3's stream must be consumed from inside the client's
        `async with` block (see `__call_bedrock_stream_async`), which yields
        events directly rather than a response dict carrying
        `ResponseMetadata` up front. This has no functional effect: `LLMfy`
        overwrites `request_call_id` on every tool call once a turn completes
        anyway (see `MessageBufferBuilder.add_assistant_message`) — it only matters if
        this method is called directly. Same pattern `GoogleAIGenerateModel`
        already uses for the same reason (no natural per-turn id available).
        """
        try:
            self.__require_aioboto3()
            params = self.__build_params(messages, tools, **kwargs)
            request_id = str(uuid.uuid4())
            tools_accum: list[dict[str, Any]] = []
            tool_use: dict[str, Any] = {}

            async for chunk in self.__call_bedrock_stream_async(params):
                ai_response, tool_use = self.__process_stream_chunk(
                    chunk, tools_accum, tool_use, request_id
                )
                yield ai_response

        except Exception as e:
            if isinstance(e, LLMfyException):
                raise  # Already handled, re-raise as-is
            raise LLMfyException(str(e), raw_error=e) from e
