import re
import uuid
from collections.abc import AsyncGenerator, Callable, Generator
from typing import Any

from llmfy.exception.llmfy_exception import LLMfyException
from llmfy.llmfy_core.llms.base_ai_model import BaseAIModel
from llmfy.llmfy_core.messages.content import Content
from llmfy.llmfy_core.messages.message import Message
from llmfy.llmfy_core.messages.message_buffer_builder import MessageBufferBuilder
from llmfy.llmfy_core.messages.role import Role
from llmfy.llmfy_core.messages.tool_call import ToolCall
from llmfy.llmfy_core.responses.ai_response import AIResponse
from llmfy.llmfy_core.responses.generation_response import GenerationResponse
from llmfy.llmfy_core.tools.tool import Tool


class LLMfy:
    """
    LLMfy framework for integrating LLM-powered applications.
    """

    def __init__(
        self,
        llm: BaseAIModel,
        system_message: str | None = None,
        input_variables: list[str] | None = None,
    ):
        """
        LLMfy init.

        Args:
            llm (BaseAIModel): Base LLM Model.
            system_message (Optional[str], optional): System message/prompt. Defaults to None.
            input_variables (Optional[List[str]], optional): Input variables, Required if in system message there are placeholder var `{{var_name}}`.
                Example: ["var_name_1", "var_name_2"]. Defaults to None.
        """
        self.model: BaseAIModel = llm
        self.system_message = system_message
        self.input_variables = input_variables or []
        self._tools: dict[str, Callable] = {}
        self._tool_definitions: dict[str, dict[str, Any]] = {}

        def _has_variable_placeholder(s):
            return bool(re.search(r"\{\{[^{}]+\}\}", s))

        def _extract_variable_names(s):
            pattern = r"\{\{(\w+)\}\}"
            return re.findall(pattern, s)

        # Validate that if system message has variable then input variable should not be empty
        if (
            _has_variable_placeholder(self.system_message)
            if self.system_message
            else False
        ) and not self.input_variables:
            variable_names = _extract_variable_names(self.system_message)
            raise LLMfyException(
                f"System messages have placeholder variables, so the `input_variables` should not be empty. "
                f"Missing input variables: {variable_names}. "
            )

        # Validate input variables
        if self.system_message and self.input_variables:
            # Validate that all required input variables are in kwargs
            variable_names = _extract_variable_names(self.system_message)
            missing_vars = [
                var for var in variable_names if var not in self.input_variables
            ]
            if missing_vars:
                raise LLMfyException(
                    f"Missing required input variables: {missing_vars}. "
                    f"Expected variables: {variable_names}."
                )

    def register_tool(self, funcs: list[Callable]) -> None:
        """Register a tool with this framework instance."""
        for func in funcs:
            if not hasattr(func, "_is_tool"):
                raise LLMfyException("Function must be decorated with @Tool")

            tool_def = Tool._get_tool_definition(func, self.model.backend)
            self._tools[func.__name__] = func
            self._tool_definitions[func.__name__] = tool_def

    def __get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions registered with this framework."""
        return list(self._tool_definitions.values())

    def __execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a registered tool."""
        if name not in self._tools:
            raise LLMfyException(f"Tool not found: {name}")
        return self._tools[name](**arguments)

    def __render_template(self, template: str, variables: dict) -> str:
        """
        Render system message template use replace.

        Note on f"{{{{{var}}}}}"
        This looks confusing but breaks down as:
        1. {{ → literal {
        2. {{ → literal {
        3. {var} → the variable name
        4. }} → literal }
        5. }} → literal }
        So f"{{{{{var}}}}}" with var="name" produces {{name}}.

        Args:
            template (str): _description_
            variables (dict): _description_

        Returns:
            Final template
        """
        for var, value in variables.items():
            template = template.replace(f"{{{{{var}}}}}", str(value))
        return template

    def __validate_system_message(self, **kwargs) -> str | None:
        # If we have a system message and input variables
        try:
            # TODO enhance system prompt
            final_system_message = self.system_message if self.system_message else ""
            if self.system_message and self.input_variables:
                # Validate that all required input variables are in kwargs
                missing_vars = [
                    var for var in self.input_variables if var not in kwargs
                ]
                if missing_vars:
                    raise LLMfyException(
                        f"Missing required input variables: {missing_vars}. "
                        f"Expected variables: {self.input_variables}, "
                        f"Received variables: {list(kwargs.keys())}"
                    )

                # Create a dictionary of variables from kwargs that match input_variables
                format_variables = {
                    var: kwargs.get(var)
                    for var in self.input_variables
                    if var in kwargs
                }

                final_system_message = self.__render_template(
                    self.system_message, format_variables
                )

            return final_system_message
        except KeyError as e:
            raise LLMfyException(f"Required variable {e} not found in kwargs") from e
        except Exception as e:
            raise LLMfyException(f"Error formatting system message: {str(e)}") from e

    def __prepare_invoke_history(
        self, messages_temp: MessageBufferBuilder, contents: str | list[Content], **kwargs
    ) -> None:
        """Seed the caller's freshly-created `messages_temp` with the system
        message (if any) and the user's `contents`. Shared by
        `invoke`/`invoke_with_tools` and their async (`ainvoke`/
        `ainvoke_with_tools`) counterparts."""
        # Generate using user role only if invoke
        messages = [Message(role=Role.USER, content=contents)]

        if self.system_message:
            # Validate system message
            final_system_message = self.__validate_system_message(**kwargs)

            # Add system message to history
            messages_temp.add_system_message(
                final_system_message if final_system_message else ""
            )

        # Add new messages to history
        for message in messages:
            # always ROLE == USER because invoke
            if message.role == Role.USER:
                messages_temp.add_user_message(
                    message.id,
                    message.content if message.content else "",
                )

    def __prepare_chat_history(
        self, messages_temp: MessageBufferBuilder, messages: list[Message], **kwargs
    ) -> None:
        """Replay the given `messages` into the caller's freshly-created
        `messages_temp`. Shared by `chat`/`chat_with_tools` and their async
        (`achat`/`achat_with_tools`) counterparts."""
        if self.system_message:
            # Validate system message
            final_system_message = self.__validate_system_message(**kwargs)

            # Add system message to history
            messages_temp.add_system_message(
                final_system_message if final_system_message else ""
            )

        # Add new messages to history
        for message in messages:
            if message.role == Role.USER:
                messages_temp.add_user_message(
                    message.id,
                    message.content if message.content else "",
                )
            elif message.role == Role.ASSISTANT:
                messages_temp.add_assistant_message(
                    id=message.id,
                    content=message.content,
                    tool_calls=message.tool_calls,
                )
            elif message.role == Role.TOOL:
                messages_temp.add_tool_message(
                    id=message.id,
                    request_call_id=message.request_call_id,
                    tool_call_id=(
                        message.tool_call_id if message.tool_call_id else ""
                    ),
                    name=message.name if message.name else "",
                    result=message.tool_results[0] if message.tool_results else "",
                    backend=self.model.backend,
                )

    def __run_tool_calls(
        self, messages_temp: MessageBufferBuilder, tool_calls: list[ToolCall]
    ) -> None:
        """Execute each requested tool call and append its result to
        `messages_temp`. Shared by the sync and async tool-calling loops."""
        for tool_call in tool_calls:
            result = self.__execute_tool(tool_call.name, tool_call.arguments)
            messages_temp.add_tool_message(
                id=str(uuid.uuid4()),
                request_call_id=tool_call.request_call_id,
                tool_call_id=tool_call.tool_call_id,
                name=tool_call.name,
                result=str(result),
                backend=self.model.backend,
            )

    def invoke(self, contents: str | list[Content], **kwargs) -> GenerationResponse:
        """
        Generate a response based on contents.

        Args:
            contents (str | List[Content]): Text or List of content objects to process
            **kwargs: Additional generation parameters

        Returns:
            GenerationResponse containing the generated response
        """
        try:
            messages_temp = MessageBufferBuilder()
            self.__prepare_invoke_history(messages_temp, contents, **kwargs)

            response = self.model.generate(
                messages_temp.get_messages(backend=self.model.backend),
                tools=self.__get_tool_definitions(),
            )

            messages_temp.add_assistant_message(
                id=str(uuid.uuid4()),
                content=response.content,
                tool_calls=response.tool_calls,
            )

            return GenerationResponse(
                result=response,
                messages=messages_temp.get_instance_messages(),
            )
        except Exception as e:
            if isinstance(e, LLMfyException):
                raise  # Already handled, re-raise as-is
            raise LLMfyException(str(e), raw_error=e) from e

    def invoke_with_tools(
        self,
        contents: str | list[Content],
        **kwargs,
    ) -> GenerationResponse:
        """
        Generate a response based on contents with tools results.

        Args:
            contents (str | List[Content]): Text or List of content objects to process
            **kwargs: Additional generation parameters

        Returns:
            GenerationResponse containing the generated response
        """
        try:
            messages_temp = MessageBufferBuilder()
            self.__prepare_invoke_history(messages_temp, contents, **kwargs)

            while True:
                response = self.model.generate(
                    messages_temp.get_messages(backend=self.model.backend),
                    tools=self.__get_tool_definitions(),
                )

                if response.tool_calls:
                    messages_temp.add_assistant_message(
                        id=str(uuid.uuid4()),
                        tool_calls=response.tool_calls,
                    )
                    self.__run_tool_calls(messages_temp, response.tool_calls)
                    continue

                messages_temp.add_assistant_message(
                    id=str(uuid.uuid4()),
                    content=response.content,
                    tool_calls=response.tool_calls,
                )

                return GenerationResponse(
                    result=response,
                    messages=messages_temp.get_instance_messages(),
                )
        except Exception as e:
            if isinstance(e, LLMfyException):
                raise  # Already handled, re-raise as-is
            raise LLMfyException(str(e), raw_error=e) from e

    def invoke_stream(
        self,
        contents: str | list[Content],
        **kwargs,
    ) -> Generator[GenerationResponse, Any, None]:
        """
        Generate a response based on contents.

        Example usage:
        ```python
        stream = chat.invoke_stream(contents="apa ibukota jakarta?", info=info)
        full_content = ""
        num = 0
        for chunk in stream:
            if isinstance(chunk, GenerationResponse):
                if chunk.result.content:
                    content = chunk.result.content
                    full_content += content
                    num += 1
                    print(f"chunk: {num}")
                    print(content, flush=True)
                    print("")
                    # print(content, end="", flush=True)

        print("--- full ---")
        print(full_content)
        ```

        Args:
            contents (str | List[Content]): Text or List of content objects to process
            **kwargs: Additional generation parameters

        Returns:
            stream (Generator[GenerationResponse, Any, None]):  Stream GenerationResponse containing the generated response
        """
        try:
            messages_temp = MessageBufferBuilder()

            # Generate using user role only if invoke
            messages = [Message(role=Role.USER, content=contents)]

            if self.system_message:
                # Validate system message
                final_system_message = self.__validate_system_message(**kwargs)

                # Add system message to history
                messages_temp.add_system_message(
                    final_system_message if final_system_message else ""
                )

            # Add new messages to history
            for message in messages:
                # always ROLE == USER because invoke
                if message.role == Role.USER:
                    messages_temp.add_user_message(
                        message.id,
                        message.content if message.content else "",
                    )

            stream = self.model.generate_stream(
                messages_temp.get_messages(backend=self.model.backend),
                tools=self.__get_tool_definitions(),
            )

            full_content = ""
            final_tool_calls = None

            for chunk in stream:
                if isinstance(chunk, AIResponse):
                    content = ""
                    thinking = ""
                    tool_calls = []
                    # Yield each chunk
                    if chunk.content:
                        content = chunk.content
                        full_content += content

                    if chunk.thinking:
                        thinking = chunk.thinking

                    if chunk.tool_calls:
                        tool_calls = chunk.tool_calls
                        # `tool_calls` above is reset to `[]` every
                        # iteration so each yielded chunk only reports its
                        # own tool-call delta — but that means it forgets a
                        # tool call completed on an earlier chunk once a
                        # later, tool-call-less chunk (e.g. a trailing
                        # finish-reason/usage-only chunk) arrives. Track the
                        # last fully-parsed tool_calls separately so the
                        # final assistant message below still carries it.
                        final_tool_calls = chunk.tool_calls

                    # update content, thinking and toolcalls only
                    yield GenerationResponse(
                        result=AIResponse(
                            content=content, thinking=thinking, tool_calls=tool_calls
                        ),
                        messages=[],
                    )

            messages_temp.add_assistant_message(
                id=str(uuid.uuid4()),
                content=full_content,
                tool_calls=final_tool_calls,
            )

            # update messages only
            yield GenerationResponse(
                result=AIResponse(),
                messages=messages_temp.get_instance_messages(),
            )
        except Exception as e:
            if isinstance(e, LLMfyException):
                raise  # Already handled, re-raise as-is
            raise LLMfyException(str(e), raw_error=e) from e

    def chat(self, messages: list[Message], **kwargs) -> GenerationResponse:
        """
        Generate a response based on a list of messages.

        Args:
            messages (List[Message]): List of Message objects to process
            **kwargs: Additional generation parameters

        Returns:
            GenerationResponse containing the generated response
        """
        try:
            messages_temp = MessageBufferBuilder()
            self.__prepare_chat_history(messages_temp, messages, **kwargs)

            response = self.model.generate(
                messages_temp.get_messages(backend=self.model.backend),
                tools=self.__get_tool_definitions(),
            )

            messages_temp.add_assistant_message(
                id=str(uuid.uuid4()),
                content=response.content,
                tool_calls=response.tool_calls,
            )

            return GenerationResponse(
                result=response,
                messages=messages_temp.get_instance_messages(),
            )
        except Exception as e:
            if isinstance(e, LLMfyException):
                raise  # Already handled, re-raise as-is
            raise LLMfyException(str(e), raw_error=e) from e

    def chat_with_tools(self, messages: list[Message], **kwargs) -> GenerationResponse:
        """
        Generate a response based on a list of messages with tools results.

        Args:
            messages (List[Message]): List of Message objects to process
            **kwargs: Additional generation parameters

        Returns:
            GenerationResponse containing the generated response
        """
        try:
            messages_temp = MessageBufferBuilder()
            self.__prepare_chat_history(messages_temp, messages, **kwargs)

            while True:
                response = self.model.generate(
                    messages_temp.get_messages(backend=self.model.backend),
                    tools=self.__get_tool_definitions(),
                )

                if response.tool_calls:
                    messages_temp.add_assistant_message(
                        id=str(uuid.uuid4()),
                        tool_calls=response.tool_calls,
                    )
                    self.__run_tool_calls(messages_temp, response.tool_calls)
                    continue

                messages_temp.add_assistant_message(
                    id=str(uuid.uuid4()),
                    content=response.content,
                    tool_calls=response.tool_calls,
                )

                return GenerationResponse(
                    result=response,
                    messages=messages_temp.get_instance_messages(),
                )
        except Exception as e:
            if isinstance(e, LLMfyException):
                raise  # Already handled, re-raise as-is
            raise LLMfyException(str(e), raw_error=e) from e

    def chat_stream(
        self,
        messages: list[Message],
        **kwargs,
    ) -> Generator[GenerationResponse, Any, None]:
        """
        Generate a streaming response based on a list of messages.

        Example usage:
        ```python
        messages = [Message(role=Role.USER, content="apa ibukota jakarta?")]
        stream = chat.chat_stream(messages, info=info)
        full_content = ""
        num = 0
        for chunk in stream:
            if isinstance(chunk, GenerationResponse):
                if chunk.result.content:
                    content = chunk.result.content
                    full_content += content
                    num += 1
                    print(f"chunk: {num}")
                    print(content, flush=True)
                    print("")
                    print(content, end="", flush=True)

        print("--- full ---")
        print(full_content)
        ```

        Args:
            messages (List[Message]): List of Message objects to process
            **kwargs: Additional generation parameters

        Returns:
            stream (Generator[GenerationResponse, Any, None]):  Stream GenerationResponse containing the generated response
        """
        try:
            messages_temp = MessageBufferBuilder()

            if self.system_message:
                # Validate system message
                final_system_message = self.__validate_system_message(**kwargs)

                # Add system message to history
                messages_temp.add_system_message(
                    final_system_message if final_system_message else ""
                )

            # Add new messages to history
            for message in messages:
                if message.role == Role.USER:
                    messages_temp.add_user_message(
                        message.id,
                        message.content if message.content else "",
                    )
                elif message.role == Role.ASSISTANT:
                    messages_temp.add_assistant_message(
                        id=message.id,
                        content=message.content,
                        tool_calls=message.tool_calls,
                    )
                elif message.role == Role.TOOL:
                    messages_temp.add_tool_message(
                        id=message.id,
                        request_call_id=message.request_call_id,
                        tool_call_id=(
                            message.tool_call_id if message.tool_call_id else ""
                        ),
                        name=message.name if message.name else "",
                        result=message.tool_results[0] if message.tool_results else "",
                        backend=self.model.backend,
                    )

            stream = self.model.generate_stream(
                messages_temp.get_messages(backend=self.model.backend),
                tools=self.__get_tool_definitions(),
            )

            full_content = ""
            final_tool_calls = None

            for chunk in stream:
                if isinstance(chunk, AIResponse):
                    content = ""
                    thinking = ""
                    tool_calls = []
                    # Yield each chunk
                    if chunk.content:
                        content = chunk.content
                        full_content += content

                    if chunk.thinking:
                        thinking = chunk.thinking

                    if chunk.tool_calls:
                        tool_calls = chunk.tool_calls
                        # `tool_calls` above is reset to `[]` every
                        # iteration so each yielded chunk only reports its
                        # own tool-call delta — but that means it forgets a
                        # tool call completed on an earlier chunk once a
                        # later, tool-call-less chunk (e.g. a trailing
                        # finish-reason/usage-only chunk) arrives. Track the
                        # last fully-parsed tool_calls separately so the
                        # final assistant message below still carries it.
                        final_tool_calls = chunk.tool_calls

                    # update content, thinking and toolcalls only
                    yield GenerationResponse(
                        result=AIResponse(
                            content=content, thinking=thinking, tool_calls=tool_calls
                        ),
                        messages=[],
                    )

            messages_temp.add_assistant_message(
                id=str(uuid.uuid4()),
                content=full_content,
                tool_calls=final_tool_calls,
            )

            # update messages only
            yield GenerationResponse(
                result=AIResponse(),
                messages=messages_temp.get_instance_messages(),
            )
        except Exception as e:
            if isinstance(e, LLMfyException):
                raise  # Already handled, re-raise as-is
            raise LLMfyException(str(e), raw_error=e) from e

    # ------------------------------------------------------------------
    # Async wrappers
    #
    # `ainvoke`/`ainvoke_with_tools`/`achat`/`achat_with_tools`/
    # `ainvoke_stream`/`achat_stream` all call the model's native async
    # method (`agenerate`/`agenerate_stream`) — a real non-blocking await
    # for backends whose SDK ships a native async client (OpenAI, Anthropic,
    # Google AI as of this writing; see each model's `agenerate`/
    # `agenerate_stream` override), falling back to a thread-offloaded sync
    # call for backends that don't (Bedrock has no async boto3 without the
    # extra `aioboto3` dependency — see `BaseAIModel.agenerate`). Either way
    # there's no `asyncio.to_thread` thread-pool ceiling (default ~32
    # workers) capping how many of these can run concurrently — matters if
    # you fan out to many LLMfy instances at once, e.g.
    # `await asyncio.gather(llm_a.ainvoke(...), llm_b.ainvoke(...))`.
    #
    # `ainvoke_stream`/`achat_stream` duplicate `invoke_stream`/`chat_stream`'s
    # accumulation loop rather than sharing it, matching this file's existing
    # sync/async duplication for the non-streaming methods above — `for` vs
    # `async for` over a differently-typed stream (`generate_stream` returns
    # a sync `Generator`, `agenerate_stream` an `AsyncGenerator`) can't share
    # one loop body. Keep any fix to the accumulation logic (e.g. how
    # `tool_calls` survives a trailing empty chunk) in sync with all four.
    #
    # Every call (sync or async) builds its own local `MessageBufferBuilder` (see
    # `invoke`/`chat` above) instead of sharing one on `self`, so concurrent
    # calls on the SAME LLMfy instance no longer race on message history.
    # `self.model` and `self._tools`/`self._tool_definitions` (populated
    # once via `register_tool`, read-only afterwards) are still shared
    # across concurrent calls — safe as long as the underlying
    # `BaseAIModel.generate`/`agenerate` is itself safe to call
    # concurrently, which is a property of the model layer, not this class.
    # ------------------------------------------------------------------

    async def ainvoke(self, contents: str | list[Content], **kwargs) -> GenerationResponse:
        """Async version of `invoke`. See `invoke` for behavior/args."""
        try:
            messages_temp = MessageBufferBuilder()
            self.__prepare_invoke_history(messages_temp, contents, **kwargs)

            response = await self.model.agenerate(
                messages_temp.get_messages(backend=self.model.backend),
                tools=self.__get_tool_definitions(),
            )

            messages_temp.add_assistant_message(
                id=str(uuid.uuid4()),
                content=response.content,
                tool_calls=response.tool_calls,
            )

            return GenerationResponse(
                result=response,
                messages=messages_temp.get_instance_messages(),
            )
        except Exception as e:
            if isinstance(e, LLMfyException):
                raise  # Already handled, re-raise as-is
            raise LLMfyException(str(e), raw_error=e) from e

    async def ainvoke_with_tools(
        self, contents: str | list[Content], **kwargs
    ) -> GenerationResponse:
        """Async version of `invoke_with_tools`. See `invoke_with_tools` for behavior/args."""
        try:
            messages_temp = MessageBufferBuilder()
            self.__prepare_invoke_history(messages_temp, contents, **kwargs)

            while True:
                response = await self.model.agenerate(
                    messages_temp.get_messages(backend=self.model.backend),
                    tools=self.__get_tool_definitions(),
                )

                if response.tool_calls:
                    messages_temp.add_assistant_message(
                        id=str(uuid.uuid4()),
                        tool_calls=response.tool_calls,
                    )
                    # Tool functions themselves stay sync (arbitrary user code —
                    # not every tool implementation is async-safe to await).
                    self.__run_tool_calls(messages_temp, response.tool_calls)
                    continue

                messages_temp.add_assistant_message(
                    id=str(uuid.uuid4()),
                    content=response.content,
                    tool_calls=response.tool_calls,
                )

                return GenerationResponse(
                    result=response,
                    messages=messages_temp.get_instance_messages(),
                )
        except Exception as e:
            if isinstance(e, LLMfyException):
                raise  # Already handled, re-raise as-is
            raise LLMfyException(str(e), raw_error=e) from e

    async def ainvoke_stream(
        self, contents: str | list[Content], **kwargs
    ) -> AsyncGenerator[GenerationResponse, Any]:
        """Async version of `invoke_stream` — uses `self.model.agenerate_stream`
        natively (no thread offload). See `invoke_stream` for behavior/args.

        Example usage:
        ```python
        async for chunk in chat.ainvoke_stream("apa ibukota jakarta?"):
            if chunk.result.content:
                print(chunk.result.content, end="", flush=True)
        ```
        """
        try:
            messages_temp = MessageBufferBuilder()
            self.__prepare_invoke_history(messages_temp, contents, **kwargs)

            stream = self.model.agenerate_stream(
                messages_temp.get_messages(backend=self.model.backend),
                tools=self.__get_tool_definitions(),
            )

            full_content = ""
            final_tool_calls = None

            async for chunk in stream:
                if isinstance(chunk, AIResponse):
                    content = ""
                    thinking = ""
                    tool_calls = []
                    # Yield each chunk
                    if chunk.content:
                        content = chunk.content
                        full_content += content

                    if chunk.thinking:
                        thinking = chunk.thinking

                    if chunk.tool_calls:
                        tool_calls = chunk.tool_calls
                        # See the matching comment in `invoke_stream` — keep
                        # a separate accumulator for the final assistant
                        # message so a tool call completed on one chunk
                        # isn't erased by a later, tool-call-less chunk.
                        final_tool_calls = chunk.tool_calls

                    # update content, thinking and toolcalls only
                    yield GenerationResponse(
                        result=AIResponse(
                            content=content, thinking=thinking, tool_calls=tool_calls
                        ),
                        messages=[],
                    )

            messages_temp.add_assistant_message(
                id=str(uuid.uuid4()),
                content=full_content,
                tool_calls=final_tool_calls,
            )

            # update messages only
            yield GenerationResponse(
                result=AIResponse(),
                messages=messages_temp.get_instance_messages(),
            )
        except Exception as e:
            if isinstance(e, LLMfyException):
                raise  # Already handled, re-raise as-is
            raise LLMfyException(str(e), raw_error=e) from e

    async def achat(self, messages: list[Message], **kwargs) -> GenerationResponse:
        """Async version of `chat`. See `chat` for behavior/args."""
        try:
            messages_temp = MessageBufferBuilder()
            self.__prepare_chat_history(messages_temp, messages, **kwargs)

            response = await self.model.agenerate(
                messages_temp.get_messages(backend=self.model.backend),
                tools=self.__get_tool_definitions(),
            )

            messages_temp.add_assistant_message(
                id=str(uuid.uuid4()),
                content=response.content,
                tool_calls=response.tool_calls,
            )

            return GenerationResponse(
                result=response,
                messages=messages_temp.get_instance_messages(),
            )
        except Exception as e:
            if isinstance(e, LLMfyException):
                raise  # Already handled, re-raise as-is
            raise LLMfyException(str(e), raw_error=e) from e

    async def achat_with_tools(
        self, messages: list[Message], **kwargs
    ) -> GenerationResponse:
        """Async version of `chat_with_tools`. See `chat_with_tools` for behavior/args."""
        try:
            messages_temp = MessageBufferBuilder()
            self.__prepare_chat_history(messages_temp, messages, **kwargs)

            while True:
                response = await self.model.agenerate(
                    messages_temp.get_messages(backend=self.model.backend),
                    tools=self.__get_tool_definitions(),
                )

                if response.tool_calls:
                    messages_temp.add_assistant_message(
                        id=str(uuid.uuid4()),
                        tool_calls=response.tool_calls,
                    )
                    # Tool functions themselves stay sync (arbitrary user code —
                    # not every tool implementation is async-safe to await).
                    self.__run_tool_calls(messages_temp, response.tool_calls)
                    continue

                messages_temp.add_assistant_message(
                    id=str(uuid.uuid4()),
                    content=response.content,
                    tool_calls=response.tool_calls,
                )

                return GenerationResponse(
                    result=response,
                    messages=messages_temp.get_instance_messages(),
                )
        except Exception as e:
            if isinstance(e, LLMfyException):
                raise  # Already handled, re-raise as-is
            raise LLMfyException(str(e), raw_error=e) from e

    async def achat_stream(
        self, messages: list[Message], **kwargs
    ) -> AsyncGenerator[GenerationResponse, Any]:
        """Async version of `chat_stream` — uses `self.model.agenerate_stream`
        natively (no thread offload). See `chat_stream` for behavior/args."""
        try:
            messages_temp = MessageBufferBuilder()
            self.__prepare_chat_history(messages_temp, messages, **kwargs)

            stream = self.model.agenerate_stream(
                messages_temp.get_messages(backend=self.model.backend),
                tools=self.__get_tool_definitions(),
            )

            full_content = ""
            final_tool_calls = None

            async for chunk in stream:
                if isinstance(chunk, AIResponse):
                    content = ""
                    thinking = ""
                    tool_calls = []
                    # Yield each chunk
                    if chunk.content:
                        content = chunk.content
                        full_content += content

                    if chunk.thinking:
                        thinking = chunk.thinking

                    if chunk.tool_calls:
                        tool_calls = chunk.tool_calls
                        # See the matching comment in `chat_stream` — keep a
                        # separate accumulator for the final assistant
                        # message so a tool call completed on one chunk
                        # isn't erased by a later, tool-call-less chunk.
                        final_tool_calls = chunk.tool_calls

                    # update content, thinking and toolcalls only
                    yield GenerationResponse(
                        result=AIResponse(
                            content=content, thinking=thinking, tool_calls=tool_calls
                        ),
                        messages=[],
                    )

            messages_temp.add_assistant_message(
                id=str(uuid.uuid4()),
                content=full_content,
                tool_calls=final_tool_calls,
            )

            # update messages only
            yield GenerationResponse(
                result=AIResponse(),
                messages=messages_temp.get_instance_messages(),
            )
        except Exception as e:
            if isinstance(e, LLMfyException):
                raise  # Already handled, re-raise as-is
            raise LLMfyException(str(e), raw_error=e) from e
