"""Unit tests for llmfy/llmfy_core/llmfy.py — the main LLMfy class.

Uses `FakeAIModel` (tests/conftest.py) as a test double for `BaseAIModel`, so
these tests never touch a real provider. `FakeAIModel` defaults to
`ModelBackend.OPENAI_CHAT`, which means the real `OpenAIChatFormatter` runs
under the hood — deliberate, since formatting itself is out of scope here
(see the per-provider formatter test files) and OpenAI's formatter has the
simplest (non-merging) tool-result behavior.
"""

import pytest

from llmfy.exception.llmfy_exception import LLMfyException
from llmfy.llmfy_core.llmfy import LLMfy
from llmfy.llmfy_core.messages.message import Message
from llmfy.llmfy_core.messages.role import Role
from llmfy.llmfy_core.messages.tool_call import ToolCall
from llmfy.llmfy_core.responses.ai_response import AIResponse
from llmfy.llmfy_core.tools.tool import Tool
from tests.conftest import FakeAIModel

# ---------------------------------------------------------------------------
# Construction / system message template validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_basic_construction(self, fake_model: FakeAIModel):
        llm = LLMfy(fake_model)
        assert llm.model is fake_model
        assert llm.input_variables == []

    def test_system_message_without_placeholders_needs_no_input_variables(
        self, fake_model
    ):
        llm = LLMfy(fake_model, system_message="You are a helpful assistant.")
        assert llm.system_message == "You are a helpful assistant."

    def test_placeholder_without_input_variables_raises(self, fake_model):
        with pytest.raises(LLMfyException, match="Missing input variables"):
            LLMfy(fake_model, system_message="You are a {{persona}}.")

    def test_placeholder_with_matching_input_variables_succeeds(self, fake_model):
        llm = LLMfy(
            fake_model,
            system_message="You are a {{persona}}.",
            input_variables=["persona"],
        )
        assert llm.input_variables == ["persona"]

    def test_placeholder_missing_from_input_variables_raises(self, fake_model):
        with pytest.raises(LLMfyException, match="Missing required input variables"):
            LLMfy(
                fake_model,
                system_message="You are a {{persona}} in {{city}}.",
                input_variables=["persona"],
            )


# ---------------------------------------------------------------------------
# register_tool
# ---------------------------------------------------------------------------


class TestRegisterTool:
    def test_registers_decorated_tool(self, fake_model):
        @Tool()
        def get_weather(location: str) -> str:
            """Get weather.

            Args:
                location (str): the city.
            """
            return "sunny"

        llm = LLMfy(fake_model)
        llm.register_tool([get_weather])
        assert "get_weather" in llm._tools

    def test_undecorated_function_raises(self, fake_model):
        llm = LLMfy(fake_model)
        with pytest.raises(LLMfyException, match="must be decorated with @Tool"):
            llm.register_tool([lambda x: x])


# ---------------------------------------------------------------------------
# invoke / chat: no tool execution even if tool_calls are returned
# ---------------------------------------------------------------------------


class TestInvoke:
    def test_returns_generation_response_with_content(self):
        model = FakeAIModel(responses=[AIResponse(content="hello!")])
        llm = LLMfy(model)
        result = llm.invoke("hi")
        assert result.result.content == "hello!"
        assert len(result.messages) == 2  # user + assistant

    def test_does_not_execute_tool_calls_even_if_returned(self):
        tool_call = ToolCall(
            tool_call_id="c1", request_call_id="", name="get_weather", arguments={}
        )
        model = FakeAIModel(responses=[AIResponse(tool_calls=[tool_call])])
        llm = LLMfy(model)

        executed = {"called": False}

        @Tool()
        def get_weather() -> str:
            """Get weather."""
            executed["called"] = True
            return "sunny"

        llm.register_tool([get_weather])
        result = llm.invoke("weather?")
        assert executed["called"] is False
        assert result.result.tool_calls[0].name == "get_weather"  # type: ignore

    def test_generic_model_exception_wrapped_as_llmfy_exception(self):
        class BrokenModel(FakeAIModel):
            def generate(self, messages, tools=None, **kwargs):
                raise RuntimeError("boom")

        llm = LLMfy(BrokenModel())
        with pytest.raises(LLMfyException, match="boom"):
            llm.invoke("hi")

    def test_llmfy_exception_from_model_is_reraised_as_is(self):
        class BrokenModel(FakeAIModel):
            def generate(self, messages, tools=None, **kwargs):
                raise LLMfyException("specific failure", status_code=418)

        llm = LLMfy(BrokenModel())
        with pytest.raises(LLMfyException) as exc_info:
            llm.invoke("hi")
        assert exc_info.value.status_code == 418

    def test_system_message_template_rendered_at_call_time(self):
        model = FakeAIModel(responses=[AIResponse(content="ok")])
        llm = LLMfy(
            model, system_message="You are a {{persona}}.", input_variables=["persona"]
        )
        llm.invoke("hi", persona="pirate")
        formatted_first_call = model.calls[0]
        system_entry = next(m for m in formatted_first_call if m["role"] == "system")
        assert system_entry["content"] == "You are a pirate."

    def test_missing_required_kwarg_at_call_time_raises(self):
        model = FakeAIModel(responses=[AIResponse(content="ok")])
        llm = LLMfy(
            model, system_message="You are a {{persona}}.", input_variables=["persona"]
        )
        with pytest.raises(LLMfyException, match="Missing required input variables"):
            llm.invoke("hi")  # persona kwarg not supplied


class TestInvokeWithTools:
    def test_stops_after_a_round_with_no_more_tool_calls(self):
        model = FakeAIModel(
            responses=[
                AIResponse(
                    tool_calls=[
                        ToolCall(
                            tool_call_id="c1",
                            request_call_id="",
                            name="get_weather",
                            arguments={},
                        )
                    ]
                ),
                AIResponse(content="It is sunny."),
            ]
        )
        llm = LLMfy(model)

        @Tool()
        def get_weather() -> str:
            """Get weather."""
            return "sunny"

        llm.register_tool([get_weather])
        result = llm.invoke_with_tools("weather?")
        assert result.result.content == "It is sunny."
        assert len(model.calls) == 2  # one round-trip with tools, one final

    def test_executes_tool_and_stringifies_result(self):
        model = FakeAIModel(
            responses=[
                AIResponse(
                    tool_calls=[
                        ToolCall(
                            tool_call_id="c1",
                            request_call_id="",
                            name="add",
                            arguments={"a": 2, "b": 3},
                        )
                    ]
                ),
                AIResponse(content="done"),
            ]
        )
        llm = LLMfy(model)

        @Tool()
        def add(a: int, b: int) -> int:
            """Add.

            Args:
                a (int): first.
                b (int): second.
            """
            return a + b

        llm.register_tool([add])
        result = llm.invoke_with_tools("add 2 and 3")
        # The tool result message (str(5) == "5") must appear in history.
        tool_messages = [m for m in result.messages if m.role == Role.TOOL]
        assert tool_messages[0].tool_results == ["5"]

    def test_unregistered_tool_call_raises(self):
        model = FakeAIModel(
            responses=[
                AIResponse(
                    tool_calls=[
                        ToolCall(
                            tool_call_id="c1",
                            request_call_id="",
                            name="mystery_tool",
                            arguments={},
                        )
                    ]
                )
            ]
        )
        llm = LLMfy(model)
        with pytest.raises(LLMfyException, match="Tool not found"):
            llm.invoke_with_tools("do something")

    def test_multi_round_tool_calling_terminates(self):
        # Two separate tool-call rounds before a final content answer —
        # proves the `while True` loop advances and does eventually stop.
        model = FakeAIModel(
            responses=[
                AIResponse(
                    tool_calls=[
                        ToolCall(
                            tool_call_id="c1",
                            request_call_id="",
                            name="step",
                            arguments={},
                        )
                    ]
                ),
                AIResponse(
                    tool_calls=[
                        ToolCall(
                            tool_call_id="c2",
                            request_call_id="",
                            name="step",
                            arguments={},
                        )
                    ]
                ),
                AIResponse(content="finished"),
            ]
        )
        llm = LLMfy(model)

        @Tool()
        def step() -> str:
            """One step."""
            return "ok"

        llm.register_tool([step])
        result = llm.invoke_with_tools("go")
        assert result.result.content == "finished"
        assert len(model.calls) == 3


# ---------------------------------------------------------------------------
# chat / chat_with_tools
# ---------------------------------------------------------------------------


class TestChat:
    def test_replays_message_history_by_role(self):
        model = FakeAIModel(responses=[AIResponse(content="final answer")])
        llm = LLMfy(model)
        messages = [
            Message(role=Role.USER, content="hi"),
            Message(role=Role.ASSISTANT, content="hello"),
            Message(role=Role.USER, content="how are you?"),
        ]
        result = llm.chat(messages)
        assert result.result.content == "final answer"
        # 3 replayed + 1 new assistant reply
        assert len(result.messages) == 4

    def test_only_first_tool_result_is_replayed_when_message_carries_multiple(self):
        model = FakeAIModel(responses=[AIResponse(content="ok")])
        llm = LLMfy(model)
        messages = [
            Message(role=Role.USER, content="hi"),
            Message(
                role=Role.ASSISTANT,
                tool_calls=[
                    ToolCall(
                        tool_call_id="c1", request_call_id="r1", name="fn", arguments={}
                    )
                ],
            ),
            Message(
                role=Role.TOOL,
                tool_call_id="c1",
                request_call_id="r1",
                tool_results=["first", "second"],
            ),
        ]
        llm.chat(messages)
        formatted = model.calls[0]
        tool_message = next(m for m in formatted if m.get("role") == "tool")
        assert tool_message["content"] == "first"


class TestChatWithTools:
    def test_executes_tools_across_rounds(self):
        model = FakeAIModel(
            responses=[
                AIResponse(
                    tool_calls=[
                        ToolCall(
                            tool_call_id="c1",
                            request_call_id="",
                            name="get_weather",
                            arguments={},
                        )
                    ]
                ),
                AIResponse(content="sunny today"),
            ]
        )
        llm = LLMfy(model)

        @Tool()
        def get_weather() -> str:
            """Get weather."""
            return "sunny"

        llm.register_tool([get_weather])
        result = llm.chat_with_tools([Message(role=Role.USER, content="weather?")])
        assert result.result.content == "sunny today"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestInvokeStream:
    def test_yields_content_chunks_then_a_final_message_only_sentinel(self):
        chunks = [AIResponse(content="Hel"), AIResponse(content="lo")]
        model = FakeAIModel(stream_chunks=chunks)
        llm = LLMfy(model)

        results = list(llm.invoke_stream("hi"))
        assert len(results) == 3  # 2 content chunks + 1 terminal sentinel
        assert results[0].result.content == "Hel"
        assert results[1].result.content == "lo"

        terminal = results[-1]
        assert terminal.result.content is None
        assert len(terminal.messages) == 2  # user + accumulated assistant

    def test_accumulates_full_content_into_final_assistant_message(self):
        chunks = [AIResponse(content="Hel"), AIResponse(content="lo")]
        model = FakeAIModel(stream_chunks=chunks)
        llm = LLMfy(model)

        results = list(llm.invoke_stream("hi"))
        final_messages = results[-1].messages
        assistant_message = next(m for m in final_messages if m.role == Role.ASSISTANT)
        assert assistant_message.content == "Hello"

    def test_empty_string_content_chunk_is_indistinguishable_from_no_content(self):
        # Documented edge case: mid-stream chunks default content to "" then
        # override only if truthy, so a genuinely empty-string chunk from the
        # model looks identical to a chunk that carried no content at all.
        chunks = [AIResponse(content="")]
        model = FakeAIModel(stream_chunks=chunks)
        llm = LLMfy(model)
        results = list(llm.invoke_stream("hi"))
        assert results[0].result.content == ""

    def test_tool_call_completed_mid_stream_survives_trailing_chunk(self):
        """Regression: a tool call that finishes parsing on one chunk must
        not be lost by a later chunk that carries no tool_calls of its own
        (e.g. a trailing finish-reason/usage-only chunk with empty content).
        The per-chunk `tool_calls` local is intentionally reset to `[]`
        every iteration (so each yielded chunk only reports its own delta),
        but that same variable used to be reused, unaccumulated, for the
        final assistant message built after the loop — so any chunk after
        the one that completed the tool call silently erased it."""
        tool_call = ToolCall(
            tool_call_id="c1", request_call_id="r1", name="get_weather", arguments={}
        )
        chunks = [AIResponse(tool_calls=[tool_call]), AIResponse(content="")]
        model = FakeAIModel(stream_chunks=chunks)
        llm = LLMfy(model)

        results = list(llm.invoke_stream("hi"))
        assistant_message = results[-1].messages[-1]
        assert assistant_message.tool_calls == [tool_call]


class TestChatStream:
    def test_yields_content_then_terminal_sentinel_with_history(self):
        chunks = [AIResponse(content="hi there")]
        model = FakeAIModel(stream_chunks=chunks)
        llm = LLMfy(model)

        results = list(llm.chat_stream([Message(role=Role.USER, content="hi")]))
        assert results[0].result.content == "hi there"
        assert results[-1].result.content is None
        assert len(results[-1].messages) == 2

    def test_tool_call_completed_mid_stream_survives_trailing_chunk(self):
        """Same regression as
        TestInvokeStream.test_tool_call_completed_mid_stream_survives_trailing_chunk,
        for `chat_stream`. This is what silently broke the FlowEngine
        streaming agent example (flowengine_agent_stream_example.py):
        `main`'s node consumes `achat_stream` (which wraps this method), and
        `should_continue` routes on `last_message.tool_calls` — if a
        completed tool call is dropped by a trailing chunk, `tool_calls`
        reads as falsy and the graph routes straight to END with an empty
        reply instead of looping into the `tools` node."""
        tool_call = ToolCall(
            tool_call_id="c1", request_call_id="r1", name="get_weather", arguments={}
        )
        chunks = [AIResponse(tool_calls=[tool_call]), AIResponse(content="")]
        model = FakeAIModel(stream_chunks=chunks)
        llm = LLMfy(model)

        results = list(llm.chat_stream([Message(role=Role.USER, content="hi")]))
        assistant_message = results[-1].messages[-1]
        assert assistant_message.tool_calls == [tool_call]


class TestNoHistoryLeakageBetweenCalls:
    """`LLMfy` no longer exposes `messages_temp`/`clear_messages_temp()`:
    each call builds its own local message history instead of a shared
    instance attribute (see `llmfy.py`'s "Async wrappers" comment for why —
    two concurrent calls on the same instance used to race on it). These
    tests exercise that guarantee through the public API instead: reusing
    one `LLMfy` instance for a second, unrelated call must not see the
    first call's history."""

    def test_invoke_does_not_leak_into_next_invoke(self):
        model = FakeAIModel(
            responses=[AIResponse(content="first"), AIResponse(content="second")]
        )
        llm = LLMfy(model)

        first = llm.invoke("hi")
        assert len(first.messages) == 2  # user + assistant

        second = llm.invoke("hi again")
        assert len(second.messages) == 2  # fresh history, not 4

    def test_chat_does_not_leak_into_next_chat(self):
        model = FakeAIModel(
            responses=[AIResponse(content="first"), AIResponse(content="second")]
        )
        llm = LLMfy(model)

        llm.chat([Message(role=Role.USER, content="hi")])
        second = llm.chat([Message(role=Role.USER, content="hi again")])

        assert len(second.messages) == 2  # only this call's user + assistant

    def test_messages_temp_is_not_a_public_attribute(self):
        llm = LLMfy(FakeAIModel())
        assert not hasattr(llm, "messages_temp")
        assert not hasattr(llm, "clear_messages_temp")


# ---------------------------------------------------------------------------
# Async mirrors
# ---------------------------------------------------------------------------


class TestAsyncMethods:
    @pytest.mark.asyncio
    async def test_ainvoke_returns_content(self):
        model = FakeAIModel(responses=[AIResponse(content="async hello")])
        llm = LLMfy(model)
        result = await llm.ainvoke("hi")
        assert result.result.content == "async hello"

    @pytest.mark.asyncio
    async def test_ainvoke_with_tools_executes_tool_then_stops(self):
        model = FakeAIModel(
            responses=[
                AIResponse(
                    tool_calls=[
                        ToolCall(
                            tool_call_id="c1",
                            request_call_id="",
                            name="get_weather",
                            arguments={},
                        )
                    ]
                ),
                AIResponse(content="sunny"),
            ]
        )
        llm = LLMfy(model)

        @Tool()
        def get_weather() -> str:
            """Get weather."""
            return "sunny"

        llm.register_tool([get_weather])
        result = await llm.ainvoke_with_tools("weather?")
        assert result.result.content == "sunny"

    @pytest.mark.asyncio
    async def test_achat_returns_content(self):
        model = FakeAIModel(responses=[AIResponse(content="async chat reply")])
        llm = LLMfy(model)
        result = await llm.achat([Message(role=Role.USER, content="hi")])
        assert result.result.content == "async chat reply"

    @pytest.mark.asyncio
    async def test_ainvoke_stream_yields_chunks(self):
        chunks = [AIResponse(content="a"), AIResponse(content="b")]
        model = FakeAIModel(stream_chunks=chunks)
        llm = LLMfy(model)

        results = [chunk async for chunk in llm.ainvoke_stream("hi")]
        assert [r.result.content for r in results[:2]] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_ainvoke_stream_uses_native_async_generate_stream(self):
        """Regression: `ainvoke_stream` used to thread-offload the whole
        sync `invoke_stream` method (`sync_gen_to_async`), which drove
        `model.generate_stream` — never the model's own native async
        `agenerate_stream`, even though every backend implements it (see
        `BaseAIModel.agenerate_stream`). Asserting `generate_stream` is
        never touched catches a regression back to that thread-offloaded
        wrapper."""

        class AsyncOnlyModel(FakeAIModel):
            def generate_stream(self, messages, tools=None, **kwargs):
                raise AssertionError("sync generate_stream should not be called")

        model = AsyncOnlyModel(stream_chunks=[AIResponse(content="a")])
        llm = LLMfy(model)

        results = [chunk async for chunk in llm.ainvoke_stream("hi")]
        assert results[0].result.content == "a"

    @pytest.mark.asyncio
    async def test_achat_stream_yields_chunks(self):
        chunks = [AIResponse(content="x")]
        model = FakeAIModel(stream_chunks=chunks)
        llm = LLMfy(model)

        results = [
            chunk
            async for chunk in llm.achat_stream([Message(role=Role.USER, content="hi")])
        ]
        assert results[0].result.content == "x"

    @pytest.mark.asyncio
    async def test_achat_stream_uses_native_async_generate_stream(self):
        """Same regression as
        TestAsyncMethods.test_ainvoke_stream_uses_native_async_generate_stream,
        for `achat_stream`."""

        class AsyncOnlyModel(FakeAIModel):
            def generate_stream(self, messages, tools=None, **kwargs):
                raise AssertionError("sync generate_stream should not be called")

        model = AsyncOnlyModel(stream_chunks=[AIResponse(content="x")])
        llm = LLMfy(model)

        results = [
            chunk
            async for chunk in llm.achat_stream([Message(role=Role.USER, content="hi")])
        ]
        assert results[0].result.content == "x"

    @pytest.mark.asyncio
    async def test_ainvoke_stream_tool_call_completed_mid_stream_survives_trailing_chunk(
        self,
    ):
        """Async counterpart of
        TestInvokeStream.test_tool_call_completed_mid_stream_survives_trailing_chunk
        — `ainvoke_stream` has its own accumulation loop (an `async for`
        over `agenerate_stream`, not a wrapped sync `invoke_stream`), so the
        same trailing-chunk tool_calls bug could resurface here
        independently."""
        tool_call = ToolCall(
            tool_call_id="c1", request_call_id="r1", name="get_weather", arguments={}
        )
        chunks = [AIResponse(tool_calls=[tool_call]), AIResponse(content="")]
        model = FakeAIModel(stream_chunks=chunks)
        llm = LLMfy(model)

        results = [chunk async for chunk in llm.ainvoke_stream("hi")]
        assistant_message = results[-1].messages[-1]
        assert assistant_message.tool_calls == [tool_call]

    @pytest.mark.asyncio
    async def test_achat_stream_tool_call_completed_mid_stream_survives_trailing_chunk(
        self,
    ):
        """Async counterpart of
        TestChatStream.test_tool_call_completed_mid_stream_survives_trailing_chunk
        for `achat_stream`."""
        tool_call = ToolCall(
            tool_call_id="c1", request_call_id="r1", name="get_weather", arguments={}
        )
        chunks = [AIResponse(tool_calls=[tool_call]), AIResponse(content="")]
        model = FakeAIModel(stream_chunks=chunks)
        llm = LLMfy(model)

        results = [
            chunk
            async for chunk in llm.achat_stream([Message(role=Role.USER, content="hi")])
        ]
        assistant_message = results[-1].messages[-1]
        assert assistant_message.tool_calls == [tool_call]

    @pytest.mark.asyncio
    async def test_async_generic_exception_wrapped(self):
        class BrokenModel(FakeAIModel):
            async def agenerate(self, messages, tools=None, **kwargs):
                raise RuntimeError("async boom")

        llm = LLMfy(BrokenModel())
        with pytest.raises(LLMfyException, match="async boom"):
            await llm.ainvoke("hi")
