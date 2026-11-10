"""LLMfy with an LLMRetryPolicy: what is retried, how often, and what never is."""

import pytest

from llmfy.exception.llmfy_exception import (
    EmptyResponseException,
    InvalidRequestException,
    ModelErrorException,
    RateLimitException,
    ServiceUnavailableException,
    TimeoutException,
    TimeoutType,
)
from llmfy.llmfy_core.llmfy import LLMfy
from llmfy.llmfy_core.messages.message import Message
from llmfy.llmfy_core.messages.role import Role
from llmfy.llmfy_core.messages.tool_call import ToolCall
from llmfy.llmfy_core.responses.ai_response import AIResponse
from llmfy.llmfy_core.retry import LLMRetryPolicy, RetryEvent
from llmfy.llmfy_core.tools.tool import Tool
from tests.conftest import FakeAIModel
from tests.llmfy_core.retry.conftest import FakeClock

OK = AIResponse(content="ok")
EMPTY = AIResponse(content="")


def model_error() -> ModelErrorException:
    return ModelErrorException("model failed", 424)


def model_timeout() -> TimeoutException:
    return TimeoutException("model timeout", 408, timeout_type=TimeoutType.MODEL)


def throttled() -> RateLimitException:
    return RateLimitException("throttled", 429)


def tool_call(name: str = "lookup") -> ToolCall:
    return ToolCall(tool_call_id="t1", request_call_id="r1", name=name, arguments={"q": "x"})


# ---------------------------------------------------------------------------
# Without a policy nothing changes
# ---------------------------------------------------------------------------


class TestNoPolicy:
    def test_errors_propagate_on_the_first_call(self, clock: FakeClock):
        model = FakeAIModel(responses=[model_error(), OK])

        with pytest.raises(ModelErrorException):
            LLMfy(model).invoke("hi")

        assert len(model.calls) == 1
        assert clock.sleeps == []

    def test_empty_replies_are_returned_as_is(self, clock: FakeClock):
        model = FakeAIModel(responses=[EMPTY, OK])

        response = LLMfy(model).invoke("hi")

        assert response.result.content == ""
        assert len(model.calls) == 1


# ---------------------------------------------------------------------------
# Retryable errors and empty replies (counted against max_attempts)
# ---------------------------------------------------------------------------


class TestRetryableErrors:
    @pytest.mark.parametrize("failure", [model_error, model_timeout], ids=["424", "408-model"])
    def test_recovers_after_retryable_errors(self, clock: FakeClock, failure):
        model = FakeAIModel(responses=[failure(), failure(), OK])

        response = LLMfy(model, retry=LLMRetryPolicy()).invoke("hi")

        assert response.result.content == "ok"
        assert len(model.calls) == 3
        assert len(clock.sleeps) == 2

    def test_raises_the_last_error_once_attempts_run_out(self, clock: FakeClock):
        errors = [model_error(), model_error(), model_error()]
        model = FakeAIModel(responses=list(errors))

        with pytest.raises(ModelErrorException) as raised:
            LLMfy(model, retry=LLMRetryPolicy(max_attempts=3)).invoke("hi")

        assert raised.value is errors[-1]
        assert len(model.calls) == 3
        assert len(clock.sleeps) == 2

    @pytest.mark.parametrize(
        "error",
        [
            InvalidRequestException("bad request", 400),
            TimeoutException("read timeout", timeout_type=TimeoutType.READ),
            throttled(),
        ],
        ids=["invalid-request", "read-timeout", "throttling-without-wait"],
    )
    def test_does_not_retry_what_the_policy_leaves_to_the_sdk(self, clock: FakeClock, error):
        model = FakeAIModel(responses=[error, OK])

        with pytest.raises(type(error)):
            LLMfy(model, retry=LLMRetryPolicy()).invoke("hi")

        assert len(model.calls) == 1
        assert clock.sleeps == []

    def test_max_attempts_one_disables_retries(self, clock: FakeClock):
        model = FakeAIModel(responses=[model_error(), OK])

        with pytest.raises(ModelErrorException):
            LLMfy(model, retry=LLMRetryPolicy(max_attempts=1)).invoke("hi")

        assert len(model.calls) == 1

    def test_custom_classifier(self, clock: FakeClock):
        model = FakeAIModel(responses=[InvalidRequestException("flaky proxy", 400), OK])
        policy = LLMRetryPolicy(is_retryable=lambda e: isinstance(e, InvalidRequestException))

        response = LLMfy(model, retry=policy).invoke("hi")

        assert response.result.content == "ok"
        assert len(model.calls) == 2

    def test_each_attempt_gets_a_freshly_built_payload(self, clock: FakeClock):
        model = FakeAIModel(responses=[model_error(), OK])

        LLMfy(model, retry=LLMRetryPolicy()).invoke("hi")

        first, second = model.calls
        assert first == second
        assert first is not second


class TestEmptyReplies:
    def test_retries_empty_replies(self, clock: FakeClock):
        model = FakeAIModel(responses=[EMPTY, AIResponse(content="   "), OK])

        response = LLMfy(model, retry=LLMRetryPolicy()).invoke("hi")

        assert response.result.content == "ok"
        assert len(model.calls) == 3

    def test_returns_the_last_empty_reply_once_attempts_run_out(self, clock: FakeClock):
        model = FakeAIModel(responses=[EMPTY, EMPTY, EMPTY, OK])

        response = LLMfy(model, retry=LLMRetryPolicy(max_attempts=3)).invoke("hi")

        assert response.result.content == ""
        assert len(model.calls) == 3

    def test_can_raise_once_attempts_run_out(self, clock: FakeClock):
        model = FakeAIModel(responses=[EMPTY, EMPTY])
        policy = LLMRetryPolicy(max_attempts=2, on_empty_exhausted="raise")

        with pytest.raises(EmptyResponseException) as raised:
            LLMfy(model, retry=policy).invoke("hi")

        assert raised.value.response.content == ""
        assert len(model.calls) == 2

    def test_retry_on_empty_false_returns_the_first_empty_reply(self, clock: FakeClock):
        model = FakeAIModel(responses=[EMPTY, OK])
        policy = LLMRetryPolicy(retry_on_empty=False, on_empty_exhausted="raise")

        response = LLMfy(model, retry=policy).invoke("hi")

        assert response.result.content == ""
        assert len(model.calls) == 1

    def test_tool_call_without_text_is_not_empty(self, clock: FakeClock):
        model = FakeAIModel(responses=[AIResponse(tool_calls=[tool_call()])])

        response = LLMfy(model, retry=LLMRetryPolicy()).invoke("hi")

        assert response.result.tool_calls
        assert len(model.calls) == 1

    def test_errors_and_empty_replies_share_the_attempt_budget(self, clock: FakeClock):
        model = FakeAIModel(responses=[model_error(), EMPTY, model_error(), OK])

        with pytest.raises(ModelErrorException):
            LLMfy(model, retry=LLMRetryPolicy(max_attempts=3)).invoke("hi")

        assert len(model.calls) == 3


# ---------------------------------------------------------------------------
# Capacity errors and throttle_wait
# ---------------------------------------------------------------------------


class TestThrottleWait:
    def test_capacity_errors_are_retried_within_the_wait(self, clock: FakeClock):
        model = FakeAIModel(responses=[throttled(), ServiceUnavailableException("busy", 503), OK])

        response = LLMfy(model, retry=LLMRetryPolicy(throttle_wait=30)).invoke("hi")

        assert response.result.content == "ok"
        assert len(model.calls) == 3

    def test_gives_up_once_the_wait_runs_out(self, clock: FakeClock):
        model = FakeAIModel(responses=[throttled() for _ in range(100)])

        with pytest.raises(RateLimitException):
            LLMfy(model, retry=LLMRetryPolicy(throttle_wait=10)).invoke("hi")

        # The last sleep is clamped so the total never exceeds the wait.
        assert sum(clock.sleeps) == pytest.approx(10)
        assert all(delay <= 15 for delay in clock.sleeps)

    def test_capacity_retries_do_not_use_up_attempts(self, clock: FakeClock):
        model = FakeAIModel(
            responses=[throttled(), throttled(), throttled(), model_error(), model_error(), OK]
        )

        response = LLMfy(
            model, retry=LLMRetryPolicy(max_attempts=3, throttle_wait=60)
        ).invoke("hi")

        assert response.result.content == "ok"
        assert len(model.calls) == 6

    def test_wait_counts_from_the_first_capacity_error(self, clock: FakeClock):
        # A model error (and its backoff) before the throttling does not
        # shorten or extend the throttle wait.
        model = FakeAIModel(responses=[model_error()] + [throttled() for _ in range(100)])

        with pytest.raises(RateLimitException):
            LLMfy(
                model, retry=LLMRetryPolicy(throttle_wait=20, base_delay=1.0)
            ).invoke("hi")

        first_sleep, *throttle_sleeps = clock.sleeps
        assert first_sleep <= 1.0
        assert sum(throttle_sleeps) == pytest.approx(20)


# ---------------------------------------------------------------------------
# Tool calling: one retry scope per round trip
# ---------------------------------------------------------------------------


class TestToolCalling:
    def test_tools_are_not_rerun_when_a_later_round_trip_is_retried(self, clock: FakeClock):
        runs: list[str] = []

        @Tool()
        def lookup(q: str) -> str:
            """Look something up.

            Args:
                q (str): query.
            """
            runs.append(q)
            return "found"

        model = FakeAIModel(
            responses=[AIResponse(tool_calls=[tool_call()]), model_error(), EMPTY, OK]
        )
        llm = LLMfy(model, retry=LLMRetryPolicy())
        llm.register_tool([lookup])

        response = llm.invoke_with_tools("hi")

        assert response.result.content == "ok"
        assert runs == ["x"]
        assert len(model.calls) == 4

    def test_chat_with_tools_retries_each_round_trip(self, clock: FakeClock):
        @Tool()
        def lookup(q: str) -> str:
            """Look something up.

            Args:
                q (str): query.
            """
            return "found"

        model = FakeAIModel(
            responses=[model_error(), AIResponse(tool_calls=[tool_call()]), model_error(), OK]
        )
        llm = LLMfy(model, retry=LLMRetryPolicy())
        llm.register_tool([lookup])

        response = llm.chat_with_tools([Message(role=Role.USER, content="hi")])

        assert response.result.content == "ok"
        assert len(model.calls) == 4


# ---------------------------------------------------------------------------
# Per-instance policies and the on_retry hook
# ---------------------------------------------------------------------------


class TestPolicyScope:
    def test_instances_sharing_a_model_keep_their_own_policy(self, clock: FakeClock):
        model = FakeAIModel(responses=[model_error(), OK, model_error(), OK])
        retrying = LLMfy(model, retry=LLMRetryPolicy())
        plain = LLMfy(model)

        assert retrying.invoke("hi").result.content == "ok"
        with pytest.raises(ModelErrorException):
            plain.invoke("hi")

    def test_model_methods_are_not_patched(self, clock: FakeClock):
        model = FakeAIModel()
        generate = model.generate

        LLMfy(model, retry=LLMRetryPolicy()).invoke("hi")

        assert model.generate == generate

    def test_on_retry_receives_an_event_per_retry(self, clock: FakeClock):
        events: list[RetryEvent] = []
        model = FakeAIModel(responses=[throttled(), model_error(), EMPTY, OK])
        policy = LLMRetryPolicy(throttle_wait=30, on_retry=events.append)

        LLMfy(model, retry=policy).invoke("hi")

        assert [(e.reason, e.attempt) for e in events] == [
            ("capacity", 1),
            ("error", 1),
            ("empty", 2),
        ]
        assert all(e.model == "FakeAIModel" for e in events)
        assert events[0].wait_remaining == pytest.approx(30)
        assert isinstance(events[1].error, ModelErrorException)
        assert events[2].error is None
        assert [e.delay for e in events] == clock.sleeps
        assert not any(e.streaming for e in events)

    def test_on_retry_errors_propagate(self, clock: FakeClock):
        def broken_hook(event: RetryEvent) -> None:
            raise RuntimeError("hook failed")

        model = FakeAIModel(responses=[model_error(), OK])

        with pytest.raises(Exception, match="hook failed"):
            LLMfy(model, retry=LLMRetryPolicy(on_retry=broken_hook)).invoke("hi")

    def test_retries_are_logged(self, clock: FakeClock, caplog: pytest.LogCaptureFixture):
        model = FakeAIModel(responses=[model_error(), OK])

        with caplog.at_level("WARNING", logger="llmfy.llmfy_core.retry.retry_runner"):
            LLMfy(model, retry=LLMRetryPolicy()).invoke("hi")

        assert "[LLM Retry] FakeAIModel attempt 1/3 failed" in caplog.text


# ---------------------------------------------------------------------------
# Streaming: retried only before the first chunk
# ---------------------------------------------------------------------------


def stream_text(chunks) -> str:
    return "".join(chunk.result.content or "" for chunk in chunks)


class TestStreaming:
    def test_retries_a_stream_that_fails_before_its_first_chunk(self, clock: FakeClock):
        model = FakeAIModel(
            stream_scripts=[[throttled()], [model_error()], [AIResponse(content="he"), AIResponse(content="llo")]]
        )

        chunks = list(
            LLMfy(model, retry=LLMRetryPolicy(throttle_wait=30)).invoke_stream("hi")
        )

        assert stream_text(chunks) == "hello"
        assert len(model.calls) == 3
        assert chunks[-1].messages[-1].content == "hello"

    def test_never_retries_after_a_chunk_was_yielded(self, clock: FakeClock):
        model = FakeAIModel(
            stream_scripts=[[AIResponse(content="he"), model_error()], [AIResponse(content="hello")]]
        )
        received: list[str] = []

        with pytest.raises(ModelErrorException):
            for chunk in LLMfy(model, retry=LLMRetryPolicy()).invoke_stream("hi"):
                received.append(chunk.result.content or "")

        assert received == ["he"]
        assert len(model.calls) == 1
        assert clock.sleeps == []

    def test_retry_stream_false_disables_stream_retries(self, clock: FakeClock):
        model = FakeAIModel(stream_scripts=[[model_error()], [AIResponse(content="hello")]])

        with pytest.raises(ModelErrorException):
            list(LLMfy(model, retry=LLMRetryPolicy(retry_stream=False)).invoke_stream("hi"))

        assert len(model.calls) == 1

    def test_stream_events_are_marked_as_streaming(self, clock: FakeClock):
        events: list[RetryEvent] = []
        model = FakeAIModel(stream_scripts=[[model_error()], [AIResponse(content="ok")]])

        list(
            LLMfy(model, retry=LLMRetryPolicy(on_retry=events.append)).chat_stream(
                [Message(role=Role.USER, content="hi")]
            )
        )

        assert [e.streaming for e in events] == [True]


# ---------------------------------------------------------------------------
# Async methods
# ---------------------------------------------------------------------------


class TestAsync:
    async def test_ainvoke_retries_and_sleeps_asynchronously(self, clock: FakeClock):
        model = FakeAIModel(responses=[model_error(), EMPTY, OK])

        response = await LLMfy(model, retry=LLMRetryPolicy()).ainvoke("hi")

        assert response.result.content == "ok"
        assert len(model.calls) == 3
        assert len(clock.sleeps) == 2

    async def test_achat_waits_out_throttling(self, clock: FakeClock):
        model = FakeAIModel(responses=[throttled(), throttled(), OK])

        response = await LLMfy(
            model, retry=LLMRetryPolicy(throttle_wait=30)
        ).achat([Message(role=Role.USER, content="hi")])

        assert response.result.content == "ok"
        assert len(model.calls) == 3

    async def test_ainvoke_with_tools_does_not_rerun_tools(self, clock: FakeClock):
        runs: list[str] = []

        @Tool()
        def lookup(q: str) -> str:
            """Look something up.

            Args:
                q (str): query.
            """
            runs.append(q)
            return "found"

        model = FakeAIModel(responses=[AIResponse(tool_calls=[tool_call()]), model_error(), OK])
        llm = LLMfy(model, retry=LLMRetryPolicy())
        llm.register_tool([lookup])

        response = await llm.ainvoke_with_tools("hi")

        assert response.result.content == "ok"
        assert runs == ["x"]

    async def test_ainvoke_stream_retries_only_before_the_first_chunk(self, clock: FakeClock):
        model = FakeAIModel(
            stream_scripts=[
                [model_error()],
                [AIResponse(content="he"), model_error()],
                [AIResponse(content="hello")],
            ]
        )
        received: list[str] = []

        with pytest.raises(ModelErrorException):
            async for chunk in LLMfy(model, retry=LLMRetryPolicy()).ainvoke_stream("hi"):
                received.append(chunk.result.content or "")

        assert received == ["he"]
        assert len(model.calls) == 2
        assert len(clock.sleeps) == 1

    async def test_achat_stream_recovers(self, clock: FakeClock):
        model = FakeAIModel(stream_scripts=[[throttled()], [AIResponse(content="ok")]])

        chunks = [
            chunk
            async for chunk in LLMfy(
                model, retry=LLMRetryPolicy(throttle_wait=30)
            ).achat_stream([Message(role=Role.USER, content="hi")])
        ]

        assert stream_text(chunks) == "ok"
        assert len(model.calls) == 2
