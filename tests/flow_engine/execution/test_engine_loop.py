"""Unit tests for llmfy/flow_engine/execution/engine_loop.py — the core
execution rewrite. These are the most important tests in the whole
rewrite: they prove the concurrency bug is fixed, fan-out/fan-in works
with order-independent reducers, the step-limit guard trips, retry/timeout
behave as configured, and hooks fire.
"""

import asyncio
import dataclasses

import pytest

from llmfy.exception.llmfy_exception import (
    CheckpointDeserializationException,
    GraphValidationException,
    NodeExecutionException,
    NodeTimeoutException,
    StepLimitExceededException,
)
from llmfy.flow_engine.checkpointer.in_memory_checkpointer import InMemoryCheckpointer
from llmfy.flow_engine.checkpointer.serde import TypeRegistry
from llmfy.flow_engine.edge.edge import Edge
from llmfy.flow_engine.execution.context import ExecutionContext
from llmfy.flow_engine.execution.engine_loop import EngineLoop, InternalEvent
from llmfy.flow_engine.execution.hooks import FlowEngineHooks
from llmfy.flow_engine.execution.policy import RetryPolicy
from llmfy.flow_engine.execution.send import Send
from llmfy.flow_engine.graph.graph_builder import build_graph
from llmfy.flow_engine.node.node import END, START, Node, NodeType
from llmfy.flow_engine.stream.node_stream_response import (
    NodeStreamResponse,
    NodeStreamType,
)


def make_nodes(**funcs) -> dict[str, Node]:
    nodes = {
        START: Node(name=START, node_type=NodeType.START),
        END: Node(name=END, node_type=NodeType.END),
    }
    for name, spec in funcs.items():
        if isinstance(spec, tuple):
            func, extra = spec
        else:
            func, extra = spec, {}
        nodes[name] = Node(name=name, node_type=NodeType.FUNCTION, func=func, **extra)
    return nodes


async def drain(agen) -> list[InternalEvent]:
    return [event async for event in agen]


class TestLinearExecution:
    async def test_state_updates_flow_through_reducers_and_replacement(self):
        async def a(state):
            return {"count": 1}

        async def b(state):
            return {"count": 1}

        nodes = make_nodes(a=a, b=b)
        edges = [Edge(START, "a"), Edge("a", "b"), Edge("b", END)]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(
            nodes, graph, reducers={"count": lambda old, new: (old or 0) + new}
        )

        ctx = ExecutionContext(session_id="s1", state={})
        events = await drain(loop.run(ctx, "a"))

        assert ctx.state == {"count": 2}
        assert events[0].type == "start"
        assert [e.node for e in events if e.type == "node_result"] == ["a", "b"]

    async def test_step_counter_increments_per_node(self):
        async def a(state):
            return {}

        async def b(state):
            return {}

        nodes = make_nodes(a=a, b=b)
        edges = [Edge(START, "a"), Edge("a", "b"), Edge("b", END)]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "a"))

        assert ctx.step == 2


class TestFanOutFanIn:
    async def test_diamond_graph_join_runs_exactly_once(self):
        combine_calls = []

        async def fetch(state):
            return {}

        async def branch_a(state):
            await asyncio.sleep(0.01)
            return {"visited": {"a"}}

        async def branch_b(state):
            return {"visited": {"b"}}

        async def combine(state):
            combine_calls.append(dict(state))
            return {}

        nodes = make_nodes(
            fetch=fetch, branch_a=branch_a, branch_b=branch_b, combine=combine
        )
        edges = [
            Edge(START, "fetch"),
            Edge("fetch", ["branch_a", "branch_b"]),
            Edge("branch_a", "combine"),
            Edge("branch_b", "combine"),
            Edge("combine", END),
        ]
        graph = build_graph(nodes, edges)
        reducers = {"visited": lambda old, new: (old or set()) | set(new)}
        loop = EngineLoop(nodes, graph, reducers=reducers)  # type: ignore

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "fetch"))

        assert len(combine_calls) == 1
        assert ctx.state["visited"] == {"a", "b"}

    async def test_join_sees_both_branches_regardless_of_completion_order(self):
        async def fetch(state):
            return {}

        async def branch_slow(state):
            await asyncio.sleep(0.02)
            return {"visited": {"slow"}}

        async def branch_fast(state):
            return {"visited": {"fast"}}

        async def combine(state):
            return {}

        nodes = make_nodes(
            fetch=fetch,
            branch_slow=branch_slow,
            branch_fast=branch_fast,
            combine=combine,
        )
        edges = [
            Edge(START, "fetch"),
            Edge("fetch", ["branch_slow", "branch_fast"]),
            Edge("branch_slow", "combine"),
            Edge("branch_fast", "combine"),
            Edge("combine", END),
        ]
        graph = build_graph(nodes, edges)
        reducers = {"visited": lambda old, new: (old or set()) | set(new)}
        loop = EngineLoop(nodes, graph, reducers=reducers)  # type: ignore

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "fetch"))

        assert ctx.state["visited"] == {"slow", "fast"}

    async def test_branches_execute_concurrently_not_sequentially(self):
        order = []

        async def fetch(state):
            return {}

        async def branch_a(state):
            order.append("a-start")
            await asyncio.sleep(0.02)
            order.append("a-end")
            return {}

        async def branch_b(state):
            order.append("b-start")
            await asyncio.sleep(0.01)
            order.append("b-end")
            return {}

        async def combine(state):
            return {}

        nodes = make_nodes(
            fetch=fetch, branch_a=branch_a, branch_b=branch_b, combine=combine
        )
        edges = [
            Edge(START, "fetch"),
            Edge("fetch", ["branch_a", "branch_b"]),
            Edge("branch_a", "combine"),
            Edge("branch_b", "combine"),
            Edge("combine", END),
        ]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "fetch"))

        # If branches ran sequentially, order would be a-start,a-end,b-start,b-end.
        # Concurrently, b (shorter sleep) finishes before a.
        assert order == ["a-start", "b-start", "b-end", "a-end"]


class TestConcurrencySafety:
    async def test_two_concurrent_runs_on_one_engine_loop_do_not_corrupt_each_other(
        self,
    ):
        async def work(state):
            await asyncio.sleep(0.01)
            return {"session": state.get("seed")}

        nodes = make_nodes(work=work)
        edges = [Edge(START, "work"), Edge("work", END)]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        ctx_a = ExecutionContext(session_id="a", state={"seed": "a"})
        ctx_b = ExecutionContext(session_id="b", state={"seed": "b"})

        async def run_ctx(ctx):
            await drain(loop.run(ctx, "work"))
            return ctx

        result_a, result_b = await asyncio.gather(run_ctx(ctx_a), run_ctx(ctx_b))

        assert result_a.state == {"seed": "a", "session": "a"}
        assert result_b.state == {"seed": "b", "session": "b"}


class TestStepLimit:
    async def test_self_loop_trips_step_limit(self):
        async def looper(state):
            return {}

        def always_loop(state):
            return "looper"

        nodes = make_nodes(looper=looper)
        edges = [
            Edge(START, "looper"),
            Edge("looper", ["looper", END], condition=always_loop),
        ]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        ctx = ExecutionContext(session_id="s1", state={}, max_steps=5)
        with pytest.raises(StepLimitExceededException) as exc_info:
            await drain(loop.run(ctx, "looper"))

        assert exc_info.value.max_steps == 5
        assert exc_info.value.session_id == "s1"

    async def test_bounded_loop_under_limit_completes(self):
        async def looper(state):
            return {"count": state.get("count", 0) + 1}

        def stop_after_three(state):
            return "looper" if state.get("count", 0) < 3 else END

        nodes = make_nodes(looper=looper)
        edges = [
            Edge(START, "looper"),
            Edge("looper", ["looper", END], condition=stop_after_three),
        ]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        ctx = ExecutionContext(session_id="s1", state={}, max_steps=10)
        await drain(loop.run(ctx, "looper"))

        assert ctx.state["count"] == 3


class TestRetryPolicy:
    async def test_succeeds_after_transient_failures(self):
        attempts = {"count": 0}

        async def flaky(state):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise ValueError("transient")
            return {"ok": True}

        nodes = make_nodes(
            flaky=(
                flaky,
                {"retry": RetryPolicy(max_attempts=3, retry_on=(ValueError,))},
            )
        )
        edges = [Edge(START, "flaky"), Edge("flaky", END)]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "flaky"))

        assert attempts["count"] == 3
        assert ctx.state == {"ok": True}

    async def test_exhausted_retries_raise_node_execution_exception(self):
        async def always_fails(state):
            raise ValueError("boom")

        nodes = make_nodes(
            always_fails=(
                always_fails,
                {"retry": RetryPolicy(max_attempts=2, retry_on=(ValueError,))},
            )
        )
        edges = [Edge(START, "always_fails"), Edge("always_fails", END)]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        ctx = ExecutionContext(session_id="s1", state={})
        with pytest.raises(NodeExecutionException) as exc_info:
            await drain(loop.run(ctx, "always_fails"))

        assert exc_info.value.node_name == "always_fails"
        assert exc_info.value.attempt == 2
        assert isinstance(exc_info.value.__cause__, ValueError)

    async def test_default_policy_is_no_retry(self):
        attempts = {"count": 0}

        async def fails_once(state):
            attempts["count"] += 1
            raise ValueError("boom")

        nodes = make_nodes(fails_once=fails_once)
        edges = [Edge(START, "fails_once"), Edge("fails_once", END)]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        ctx = ExecutionContext(session_id="s1", state={})
        with pytest.raises(NodeExecutionException):
            await drain(loop.run(ctx, "fails_once"))

        assert attempts["count"] == 1


class TestTimeout:
    async def test_node_exceeding_timeout_raises_node_timeout_exception(self):
        async def slow(state):
            await asyncio.sleep(0.05)
            return {}

        nodes = make_nodes(slow=(slow, {"timeout": 0.01}))
        edges = [Edge(START, "slow"), Edge("slow", END)]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        ctx = ExecutionContext(session_id="s1", state={})
        with pytest.raises(NodeTimeoutException) as exc_info:
            await drain(loop.run(ctx, "slow"))

        assert exc_info.value.node_name == "slow"
        assert exc_info.value.timeout_seconds == 0.01


class TestHooks:
    async def test_on_node_start_and_end_called_once_per_node(self):
        starts = []
        ends = []

        async def a(state):
            return {"x": 1}

        nodes = make_nodes(a=a)
        edges = [Edge(START, "a"), Edge("a", END)]
        graph = build_graph(nodes, edges)
        hooks = FlowEngineHooks(
            on_node_start=lambda name, state: starts.append(name),
            on_node_end=lambda name, state, updates: ends.append((name, updates)),
        )
        loop = EngineLoop(nodes, graph, reducers={}, hooks=hooks)

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "a"))

        assert starts == ["a"]
        assert ends == [("a", {"x": 1})]

    async def test_on_error_called_and_exception_still_propagates(self):
        errors = []

        async def failing(state):
            raise ValueError("boom")

        nodes = make_nodes(failing=failing)
        edges = [Edge(START, "failing"), Edge("failing", END)]
        graph = build_graph(nodes, edges)
        hooks = FlowEngineHooks(
            on_error=lambda name, exc: errors.append((name, type(exc)))
        )
        loop = EngineLoop(nodes, graph, reducers={}, hooks=hooks)

        ctx = ExecutionContext(session_id="s1", state={})
        with pytest.raises(NodeExecutionException):
            await drain(loop.run(ctx, "failing"))

        assert errors == [("failing", NodeExecutionException)]


class TestStreamingNodes:
    async def test_stream_node_yields_stream_then_result_events(self):
        async def streamer(state):
            yield NodeStreamResponse(
                type=NodeStreamType.STREAM, content="a", state=None
            )
            yield NodeStreamResponse(
                type=NodeStreamType.STREAM, content="b", state=None
            )
            yield NodeStreamResponse(
                type=NodeStreamType.RESULT, content="ab", state={"out": "ab"}
            )

        nodes = make_nodes(streamer=(streamer, {"stream": True}))
        edges = [Edge(START, "streamer"), Edge("streamer", END)]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        ctx = ExecutionContext(session_id="s1", state={})
        events = await drain(loop.run(ctx, "streamer"))

        stream_events = [e for e in events if e.type == "node_stream"]
        result_events = [e for e in events if e.type == "node_result"]
        assert [e.content for e in stream_events] == ["a", "b"]
        assert len(result_events) == 1
        assert result_events[0].content == "ab"
        assert ctx.state == {"out": "ab"}

    async def test_non_node_stream_response_raises_graph_validation_exception(self):
        async def bad_streamer(state):
            yield "not a NodeStreamResponse"

        nodes = make_nodes(bad_streamer=(bad_streamer, {"stream": True}))
        edges = [Edge(START, "bad_streamer"), Edge("bad_streamer", END)]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        ctx = ExecutionContext(session_id="s1", state={})
        with pytest.raises(GraphValidationException):
            await drain(loop.run(ctx, "bad_streamer"))


@dataclasses.dataclass
class Unregistered:
    value: str


class TestCheckpointing:
    async def test_in_memory_checkpointer_accepts_unregistered_custom_objects(self):
        # InMemoryCheckpointer.requires_serialization is False — it keeps
        # live objects via deepcopy, so no type registry is needed at all.
        async def a(state):
            return {"obj": Unregistered(value="x")}

        nodes = make_nodes(a=a)
        edges = [Edge(START, "a"), Edge("a", END)]
        graph = build_graph(nodes, edges)
        checkpointer = InMemoryCheckpointer()
        loop = EngineLoop(nodes, graph, reducers={}, checkpointer=checkpointer)

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "a"))

        saved = await checkpointer.load("s1")
        assert saved.state["obj"] == Unregistered(value="x")  # type: ignore

    async def test_external_checkpointer_requires_registered_types(self):
        class FakeExternalCheckpointer(InMemoryCheckpointer):
            requires_serialization = True

        async def a(state):
            return {"obj": Unregistered(value="x")}

        nodes = make_nodes(a=a)
        edges = [Edge(START, "a"), Edge("a", END)]
        graph = build_graph(nodes, edges)
        checkpointer = FakeExternalCheckpointer()
        loop = EngineLoop(nodes, graph, reducers={}, checkpointer=checkpointer)

        ctx = ExecutionContext(session_id="s1", state={})
        with pytest.raises(CheckpointDeserializationException):
            await drain(loop.run(ctx, "a"))

    async def test_external_checkpointer_saves_registered_type_as_safe_dict(self):
        class FakeExternalCheckpointer(InMemoryCheckpointer):
            requires_serialization = True

        async def a(state):
            return {"obj": Unregistered(value="x")}

        nodes = make_nodes(a=a)
        edges = [Edge(START, "a"), Edge("a", END)]
        graph = build_graph(nodes, edges)
        checkpointer = FakeExternalCheckpointer()
        registry = TypeRegistry(types=[Unregistered])
        loop = EngineLoop(
            nodes, graph, reducers={}, checkpointer=checkpointer, type_registry=registry
        )

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "a"))

        saved = await checkpointer.load("s1")
        assert saved.state["obj"]["__type__"].endswith("Unregistered")  # type: ignore
        assert saved.state["obj"]["data"] == {"value": "x"}  # type: ignore


class TestPrevNode:
    """`CheckpointMetadata.prev_node` records whichever node's completion
    led to `node` running — `START` for a run's very first node, the
    fan-out source for each of its static-fan-out branches, and the
    dispatching node for a dynamic (`Send`) fan-out's single checkpoint."""

    async def test_first_node_has_start_as_prev(self):
        nodes = make_nodes(a=lambda s: {})
        edges = [Edge(START, "a"), Edge("a", END)]
        graph = build_graph(nodes, edges)
        checkpointer = InMemoryCheckpointer()
        loop = EngineLoop(nodes, graph, reducers={}, checkpointer=checkpointer)

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "a"))

        saved = await checkpointer.load("s1")
        assert saved.metadata.node == "a"  # type: ignore
        assert saved.metadata.prev_node == START  # type: ignore

    async def test_linear_chain_records_actual_predecessor(self):
        nodes = make_nodes(a=lambda s: {}, b=lambda s: {})
        edges = [Edge(START, "a"), Edge("a", "b"), Edge("b", END)]
        graph = build_graph(nodes, edges)
        checkpointer = InMemoryCheckpointer()
        loop = EngineLoop(nodes, graph, reducers={}, checkpointer=checkpointer)

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "a"))

        saved = await checkpointer.list("s1", limit=100)
        by_node = {c.metadata.node: c.metadata.prev_node for c in saved}
        assert by_node["a"] == START
        assert by_node["b"] == "a"

    async def test_static_fan_out_branches_record_the_fan_out_source(self):
        nodes = make_nodes(split=lambda s: {}, left=lambda s: {}, right=lambda s: {})
        edges = [
            Edge(START, "split"),
            Edge("split", ["left", "right"]),
            Edge("left", END),
            Edge("right", END),
        ]
        graph = build_graph(nodes, edges)
        checkpointer = InMemoryCheckpointer()
        loop = EngineLoop(nodes, graph, reducers={}, checkpointer=checkpointer)

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "split"))

        saved = await checkpointer.list("s1", limit=100)
        by_node = {c.metadata.node: c.metadata.prev_node for c in saved}
        assert by_node["left"] == "split"
        assert by_node["right"] == "split"

    async def test_dynamic_fan_out_checkpoint_records_the_dispatching_node(self):
        async def process_item(state):
            return {"seen": [state["item"]]}

        def route(state):
            return [Send("process_item", {"item": i}) for i in range(3)]

        nodes = make_nodes(split=lambda s: {}, process_item=process_item)
        edges = [
            Edge(START, "split"),
            Edge("split", ["process_item"], condition=route),
            Edge("process_item", END),
        ]
        graph = build_graph(nodes, edges)
        reducers = {"seen": lambda old, new: (old or []) + new}
        checkpointer = InMemoryCheckpointer()
        loop = EngineLoop(
            nodes, graph, reducers=reducers, checkpointer=checkpointer  # type: ignore
        )

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "split"))

        saved = await checkpointer.list("s1", limit=100)
        by_node = {c.metadata.node: c.metadata.prev_node for c in saved}
        # One checkpoint for the whole dispatch, tagged with the Send
        # target ("process_item") — its prev_node is "split", the node
        # whose conditional routing produced the Sends, not any individual
        # branch (branches don't get their own checkpoints — see
        # TestDynamicFanOut below).
        assert by_node["process_item"] == "split"


class TestCheckpointExtendedMetadata:
    """`CheckpointMetadata.attempt` (the winning try, 1-indexed),
    `updated_fields` (the update's own keys), and `dispatch_id` (set only
    for a dynamic fan-out's single aggregate checkpoint)."""

    async def test_attempt_is_1_on_first_try_success(self):
        nodes = make_nodes(a=lambda s: {"x": 1})
        edges = [Edge(START, "a"), Edge("a", END)]
        graph = build_graph(nodes, edges)
        checkpointer = InMemoryCheckpointer()
        loop = EngineLoop(nodes, graph, reducers={}, checkpointer=checkpointer)

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "a"))

        saved = await checkpointer.load("s1")
        assert saved.metadata.attempt == 1  # type: ignore

    async def test_attempt_reflects_the_winning_retry(self):
        calls = {"n": 0}

        async def flaky(state):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("transient")
            return {"x": 1}

        nodes = make_nodes(
            a=(flaky, {"retry": RetryPolicy(max_attempts=5, retry_on=(ValueError,))})
        )
        edges = [Edge(START, "a"), Edge("a", END)]
        graph = build_graph(nodes, edges)
        checkpointer = InMemoryCheckpointer()
        loop = EngineLoop(nodes, graph, reducers={}, checkpointer=checkpointer)

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "a"))

        saved = await checkpointer.load("s1")
        assert saved.metadata.attempt == 3  # type: ignore

    async def test_attempt_reflects_the_winning_retry_for_a_stream_node(self):
        # `_execute_stream_node_with_policy` threads the winning attempt
        # back via a mutable `attempt_holder` box rather than a return
        # value (an async generator can't itself return one) — a distinct
        # code path from the non-streaming case above, worth covering
        # separately.
        calls = {"n": 0}

        async def flaky_streamer(state):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ValueError("transient")
            yield NodeStreamResponse(
                type=NodeStreamType.RESULT, content="ok", state={"x": 1}
            )

        nodes = make_nodes(
            a=(
                flaky_streamer,
                {
                    "stream": True,
                    "retry": RetryPolicy(max_attempts=3, retry_on=(ValueError,)),
                },
            )
        )
        edges = [Edge(START, "a"), Edge("a", END)]
        graph = build_graph(nodes, edges)
        checkpointer = InMemoryCheckpointer()
        loop = EngineLoop(nodes, graph, reducers={}, checkpointer=checkpointer)

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "a"))

        saved = await checkpointer.load("s1")
        assert saved.metadata.attempt == 2  # type: ignore

    async def test_dispatch_id_is_none_for_a_regular_checkpoint(self):
        nodes = make_nodes(a=lambda s: {"x": 1})
        edges = [Edge(START, "a"), Edge("a", END)]
        graph = build_graph(nodes, edges)
        checkpointer = InMemoryCheckpointer()
        loop = EngineLoop(nodes, graph, reducers={}, checkpointer=checkpointer)

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "a"))

        saved = await checkpointer.load("s1")
        assert saved.metadata.dispatch_id is None  # type: ignore

    async def test_updated_fields_matches_the_nodes_returned_keys(self):
        nodes = make_nodes(a=lambda s: {"x": 1, "y": 2})
        edges = [Edge(START, "a"), Edge("a", END)]
        graph = build_graph(nodes, edges)
        checkpointer = InMemoryCheckpointer()
        loop = EngineLoop(nodes, graph, reducers={}, checkpointer=checkpointer)

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "a"))

        saved = await checkpointer.load("s1")
        assert saved.metadata.updated_fields == ["x", "y"]  # type: ignore

    async def test_updated_fields_is_empty_when_node_returns_nothing(self):
        nodes = make_nodes(a=lambda s: {})
        edges = [Edge(START, "a"), Edge("a", END)]
        graph = build_graph(nodes, edges)
        checkpointer = InMemoryCheckpointer()
        loop = EngineLoop(nodes, graph, reducers={}, checkpointer=checkpointer)

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "a"))

        saved = await checkpointer.load("s1")
        assert saved.metadata.updated_fields == []  # type: ignore

    async def test_dynamic_fan_out_checkpoint_has_no_single_attempt_but_has_dispatch_id(
        self,
    ):
        async def process_item(state):
            return {"seen": [state["item"]]}

        def route(state):
            return [Send("process_item", {"item": i}) for i in range(3)]

        nodes = make_nodes(split=lambda s: {}, process_item=process_item)
        edges = [
            Edge(START, "split"),
            Edge("split", ["process_item"], condition=route),
            Edge("process_item", END),
        ]
        graph = build_graph(nodes, edges)
        reducers = {"seen": lambda old, new: (old or []) + new}
        checkpointer = InMemoryCheckpointer()
        loop = EngineLoop(
            nodes, graph, reducers=reducers, checkpointer=checkpointer  # type: ignore
        )

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "split"))

        saved = await checkpointer.list("s1", limit=100)
        fan_out_meta = next(c.metadata for c in saved if c.metadata.node == "process_item")
        # No single attempt count applies across 3 independently-retried
        # branches committed together — see TestDynamicFanOut below for
        # how this one checkpoint represents the whole dispatch.
        assert fan_out_meta.attempt is None
        assert fan_out_meta.dispatch_id is not None
        assert fan_out_meta.updated_fields == ["seen"]


class TestDynamicFanOut:
    """`Send`-based dynamic fan-out: a routing function returns a
    runtime-determined number of `Send`s instead of a single next-node
    string, all targeting one declared conditional-edge target."""

    async def test_n_item_dynamic_fan_out_runs_once_per_send_and_reduces_once(self):
        combine_calls = []
        process_calls = []

        async def process_item(state):
            process_calls.append(state["item"])
            return {"items": [state["item"]]}

        async def combine(state):
            combine_calls.append(dict(state))
            return {}

        def route(state):
            return [Send("process_item", {"item": i}) for i in range(5)]

        nodes = make_nodes(
            split=lambda s: {}, process_item=process_item, combine=combine
        )
        edges = [
            Edge(START, "split"),
            Edge("split", ["process_item"], condition=route),
            Edge("process_item", "combine"),
            Edge("combine", END),
        ]
        graph = build_graph(nodes, edges)
        reducers = {"items": lambda old, new: (old or []) + new}
        loop = EngineLoop(nodes, graph, reducers=reducers)  # type: ignore

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "split"))

        assert sorted(process_calls) == [0, 1, 2, 3, 4]
        assert sorted(ctx.state["items"]) == [0, 1, 2, 3, 4]
        assert len(combine_calls) == 1

    async def test_empty_list_of_sends_produces_no_execution_and_no_further_advance(
        self,
    ):
        process_calls = []
        combine_calls = []

        async def process_item(state):
            process_calls.append(state)
            return {}

        async def combine(state):
            combine_calls.append(state)
            return {}

        def route(state):
            return []

        nodes = make_nodes(
            split=lambda s: {}, process_item=process_item, combine=combine
        )
        edges = [
            Edge(START, "split"),
            Edge("split", ["process_item"], condition=route),
            Edge("process_item", "combine"),
            Edge("combine", END),
        ]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "split"))

        assert process_calls == []
        assert combine_calls == []

    async def test_single_bare_send_not_wrapped_in_a_list(self):
        process_calls = []

        async def process_item(state):
            process_calls.append(state["item"])
            return {}

        def route(state):
            return Send("process_item", {"item": "solo"})

        nodes = make_nodes(split=lambda s: {}, process_item=process_item)
        edges = [
            Edge(START, "split"),
            Edge("split", ["process_item"], condition=route),
            Edge("process_item", END),
        ]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "split"))

        assert process_calls == ["solo"]

    async def test_send_targeting_undeclared_node_raises_graph_validation_exception(
        self,
    ):
        def route(state):
            return [Send("not_a_declared_target", {})]

        nodes = make_nodes(split=lambda s: {}, process_item=lambda s: {})
        edges = [
            Edge(START, "split"),
            Edge("split", ["process_item"], condition=route),
            Edge("process_item", END),
        ]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        ctx = ExecutionContext(session_id="s1", state={})
        with pytest.raises(GraphValidationException, match="not_a_declared_target"):
            await drain(loop.run(ctx, "split"))

    async def test_sends_targeting_different_nodes_in_one_call_raises(self):
        def route(state):
            return [Send("process_item", {}), Send("other_item", {})]

        nodes = make_nodes(
            split=lambda s: {}, process_item=lambda s: {}, other_item=lambda s: {}
        )
        edges = [
            Edge(START, "split"),
            Edge("split", ["process_item", "other_item"], condition=route),
            Edge("process_item", END),
            Edge("other_item", END),
        ]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        ctx = ExecutionContext(session_id="s1", state={})
        with pytest.raises(GraphValidationException, match="must target the same node"):
            await drain(loop.run(ctx, "split"))

    async def test_list_with_non_send_item_raises(self):
        def route(state):
            return [Send("process_item", {}), "not-a-send"]

        nodes = make_nodes(split=lambda s: {}, process_item=lambda s: {})
        edges = [
            Edge(START, "split"),
            Edge("split", ["process_item"], condition=route),
            Edge("process_item", END),
        ]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        ctx = ExecutionContext(session_id="s1", state={})
        with pytest.raises(GraphValidationException, match="non-Send item"):
            await drain(loop.run(ctx, "split"))

    async def test_branch_state_replaces_input_not_merges_with_parent_state(self):
        seen_states = []

        async def process_item(state):
            seen_states.append(dict(state))
            return {}

        def route(state):
            return [Send("process_item", {"item": "only-this"})]

        nodes = make_nodes(split=lambda s: {}, process_item=process_item)
        edges = [
            Edge(START, "split"),
            Edge("split", ["process_item"], condition=route),
            Edge("process_item", END),
        ]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        ctx = ExecutionContext(
            session_id="s1", state={"parent_only_key": "should-not-leak"}
        )
        await drain(loop.run(ctx, "split"))

        assert seen_states == [{"item": "only-this"}]
        assert "parent_only_key" not in seen_states[0]

    async def test_branch_output_still_merges_into_shared_state(self):
        def route(state):
            return [Send("process_item", {"item": i}) for i in range(3)]

        async def process_item(state):
            return {"total": state["item"]}

        nodes = make_nodes(split=lambda s: {}, process_item=process_item)
        edges = [
            Edge(START, "split"),
            Edge("split", ["process_item"], condition=route),
            Edge("process_item", END),
        ]
        graph = build_graph(nodes, edges)
        reducers = {"total": lambda old, new: (old or 0) + new}
        loop = EngineLoop(nodes, graph, reducers=reducers)  # type: ignore

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "split"))

        assert ctx.state["total"] == 0 + 1 + 2

    async def test_hooks_and_stream_events_reflect_branch_state_not_shared_state(self):
        start_states = []

        def on_start(name, state):
            if name == "process_item":
                start_states.append(dict(state))

        async def process_item(state):
            return {}

        def route(state):
            return [Send("process_item", {"item": i}) for i in range(2)]

        nodes = make_nodes(split=lambda s: {}, process_item=process_item)
        edges = [
            Edge(START, "split"),
            Edge("split", ["process_item"], condition=route),
            Edge("process_item", END),
        ]
        graph = build_graph(nodes, edges)
        hooks = FlowEngineHooks(on_node_start=on_start)
        loop = EngineLoop(nodes, graph, reducers={}, hooks=hooks)

        ctx = ExecutionContext(session_id="s1", state={"shared": "parent"})
        await drain(loop.run(ctx, "split"))

        assert sorted(s["item"] for s in start_states) == [0, 1]
        assert all("shared" not in s for s in start_states)

    async def test_node_stream_event_reflects_branch_state_for_streaming_send_target(
        self,
    ):
        async def process_item(state):
            yield NodeStreamResponse(type=NodeStreamType.STREAM, content=state["item"])
            yield NodeStreamResponse(
                type=NodeStreamType.RESULT, content="done", state={}
            )

        def route(state):
            return [Send("process_item", {"item": "branch-only"})]

        nodes = make_nodes(
            split=lambda s: {}, process_item=(process_item, {"stream": True})
        )
        edges = [
            Edge(START, "split"),
            Edge("split", ["process_item"], condition=route),
            Edge("process_item", END),
        ]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        ctx = ExecutionContext(session_id="s1", state={"shared": "parent"})
        events = await drain(loop.run(ctx, "split"))

        stream_events = [e for e in events if e.type == "node_stream"]
        assert len(stream_events) == 1
        assert stream_events[0].state == {"item": "branch-only"}
        assert "shared" not in stream_events[0].state  # type: ignore

    async def test_step_limit_counts_each_dynamic_branch(self):
        def route(state):
            return [Send("process_item", {"item": i}) for i in range(10)]

        nodes = make_nodes(split=lambda s: {}, process_item=lambda s: {})
        edges = [
            Edge(START, "split"),
            Edge("split", ["process_item"], condition=route),
            Edge("process_item", END),
        ]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        # split (1) + 10 branches = 11 steps, exceeding max_steps=5.
        ctx = ExecutionContext(session_id="s1", state={}, max_steps=5)
        with pytest.raises(StepLimitExceededException):
            await drain(loop.run(ctx, "split"))

    async def test_retry_policy_applies_per_branch(self):
        attempts: dict[int, int] = {}

        async def process_item(state):
            item = state["item"]
            attempts[item] = attempts.get(item, 0) + 1
            if attempts[item] < 2:
                raise ValueError("transient")
            return {}

        def route(state):
            return [Send("process_item", {"item": i}) for i in range(3)]

        nodes = make_nodes(
            split=lambda s: {},
            process_item=(
                process_item,
                {"retry": RetryPolicy(max_attempts=2, retry_on=(ValueError,))},
            ),
        )
        edges = [
            Edge(START, "split"),
            Edge("split", ["process_item"], condition=route),
            Edge("process_item", END),
        ]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "split"))

        assert attempts == {0: 2, 1: 2, 2: 2}

    async def test_timeout_applies_per_branch(self):
        async def slow_item(state):
            if state["item"] == 1:
                await asyncio.sleep(0.05)
            return {}

        def route(state):
            return [Send("process_item", {"item": i}) for i in range(2)]

        nodes = make_nodes(
            split=lambda s: {}, process_item=(slow_item, {"timeout": 0.01})
        )
        edges = [
            Edge(START, "split"),
            Edge("split", ["process_item"], condition=route),
            Edge("process_item", END),
        ]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        ctx = ExecutionContext(session_id="s1", state={})
        with pytest.raises(NodeTimeoutException):
            await drain(loop.run(ctx, "split"))

    async def test_hooks_fire_once_per_branch(self):
        starts = []
        ends = []

        async def process_item(state):
            return {}

        def route(state):
            return [Send("process_item", {"item": i}) for i in range(4)]

        nodes = make_nodes(split=lambda s: {}, process_item=process_item)
        edges = [
            Edge(START, "split"),
            Edge("split", ["process_item"], condition=route),
            Edge("process_item", END),
        ]
        graph = build_graph(nodes, edges)
        hooks = FlowEngineHooks(
            on_node_start=lambda name, state: starts.append(name),
            on_node_end=lambda name, state, updates: ends.append(name),
        )
        loop = EngineLoop(nodes, graph, reducers={}, hooks=hooks)

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "split"))

        assert starts.count("process_item") == 4
        assert ends.count("process_item") == 4

    async def test_one_branch_raising_leaves_ctx_state_completely_untouched(self):
        """All-or-nothing commit: even branches that succeeded before the
        failing one must NOT be visible on `ctx.state` afterward — a
        dynamic-fan-out dispatch commits atomically or not at all."""
        applied = []

        async def process_item(state):
            item = state["item"]
            if item == 0:
                await asyncio.sleep(0.02)
            if item == 1:
                raise ValueError("boom")
            applied.append(item)
            return {"seen": [item]}

        def route(state):
            return [Send("process_item", {"item": i}) for i in range(3)]

        nodes = make_nodes(split=lambda s: {}, process_item=process_item)
        edges = [
            Edge(START, "split"),
            Edge("split", ["process_item"], condition=route),
            Edge("process_item", END),
        ]
        graph = build_graph(nodes, edges)
        reducers = {"seen": lambda old, new: (old or []) + new}
        loop = EngineLoop(nodes, graph, reducers=reducers)  # type: ignore

        ctx = ExecutionContext(session_id="s1", state={})
        with pytest.raises(NodeExecutionException):
            await drain(loop.run(ctx, "split"))

        # Branch 2 (fast, no sleep) may finish before branch 1 raises, so
        # `process_item` itself may still have appended to `applied` — but
        # none of that must ever reach shared state: the dispatch never
        # commits when any branch fails, full stop.
        assert "seen" not in ctx.state

    async def test_successful_dispatch_saves_exactly_one_checkpoint_not_one_per_branch(
        self,
    ):
        async def process_item(state):
            return {"seen": [state["item"]]}

        def route(state):
            return [Send("process_item", {"item": i}) for i in range(4)]

        async def combine(state):
            return {}

        nodes = make_nodes(
            split=lambda s: {}, process_item=process_item, combine=combine
        )
        edges = [
            Edge(START, "split"),
            Edge("split", ["process_item"], condition=route),
            Edge("process_item", "combine"),
            Edge("combine", END),
        ]
        graph = build_graph(nodes, edges)
        reducers = {"seen": lambda old, new: (old or []) + new}
        checkpointer = InMemoryCheckpointer()
        loop = EngineLoop(
            nodes,
            graph,
            reducers=reducers,  # type: ignore
            checkpointer=checkpointer,  # type: ignore
        )

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "split"))

        saved = await checkpointer.list("s1", limit=100)
        node_names = [c.metadata.node for c in saved]
        # One checkpoint for the whole dispatch (tagged with the Send
        # target, "process_item"), not one per branch, plus one for the
        # downstream "combine" node — never N+1 for N branches.
        assert node_names.count("process_item") == 1
        assert node_names.count("combine") == 1

    async def test_failed_dispatch_saves_no_checkpoint(self):
        async def process_item(state):
            if state["item"] == 1:
                raise ValueError("boom")
            return {"seen": [state["item"]]}

        def route(state):
            return [Send("process_item", {"item": i}) for i in range(3)]

        nodes = make_nodes(split=lambda s: {}, process_item=process_item)
        edges = [
            Edge(START, "split"),
            Edge("split", ["process_item"], condition=route),
            Edge("process_item", END),
        ]
        graph = build_graph(nodes, edges)
        reducers = {"seen": lambda old, new: (old or []) + new}
        checkpointer = InMemoryCheckpointer()
        loop = EngineLoop(
            nodes,
            graph,
            reducers=reducers,  # type: ignore
            checkpointer=checkpointer,  # type: ignore
        )

        ctx = ExecutionContext(session_id="s1", state={})
        with pytest.raises(NodeExecutionException):
            await drain(loop.run(ctx, "split"))

        saved = await checkpointer.list("s1", limit=100)
        assert all(c.metadata.node != "process_item" for c in saved)

    async def test_on_node_end_sees_pre_commit_state_for_a_send_branch(self):
        seen_at_end = []

        async def process_item(state):
            return {"marker": state["item"]}

        def route(state):
            return [Send("process_item", {"item": "x"})]

        nodes = make_nodes(split=lambda s: {}, process_item=process_item)
        edges = [
            Edge(START, "split"),
            Edge("split", ["process_item"], condition=route),
            Edge("process_item", END),
        ]
        graph = build_graph(nodes, edges)
        hooks = FlowEngineHooks(
            on_node_end=lambda name, state, updates: (
                seen_at_end.append(dict(state)) if name == "process_item" else None
            )
        )
        loop = EngineLoop(nodes, graph, reducers={}, hooks=hooks)

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "split"))

        assert len(seen_at_end) == 1
        # The dispatch hadn't committed yet when on_node_end fired for this
        # branch — "marker" only lands on ctx.state after the barrier.
        assert "marker" not in seen_at_end[0]
        assert ctx.state["marker"] == "x"

    async def test_branch_correlation_ids_populate_and_differ_across_dispatches(self):
        def route(state):
            return [Send("process_item", {"item": i}) for i in range(3)]

        async def process_item(state):
            return {}

        nodes = make_nodes(split=lambda s: {}, process_item=process_item)
        edges = [
            Edge(START, "split"),
            Edge("split", ["process_item"], condition=route),
            Edge("process_item", END),
        ]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        ctx1 = ExecutionContext(session_id="s1", state={})
        events1 = await drain(loop.run(ctx1, "split"))
        results1 = [
            e for e in events1 if e.type == "node_result" and e.node == "process_item"
        ]

        assert sorted(e.branch_index for e in results1) == [0, 1, 2]  # type: ignore
        assert all(e.branch_total == 3 for e in results1)
        dispatch_ids1 = {e.dispatch_id for e in results1}
        assert len(dispatch_ids1) == 1
        assert None not in dispatch_ids1

        ctx2 = ExecutionContext(session_id="s2", state={})
        events2 = await drain(loop.run(ctx2, "split"))
        results2 = [
            e for e in events2 if e.type == "node_result" and e.node == "process_item"
        ]
        dispatch_ids2 = {e.dispatch_id for e in results2}

        assert dispatch_ids1 != dispatch_ids2

    async def test_correlation_ids_are_none_outside_dynamic_fan_out(self):
        async def a(state):
            return {}

        async def branch_a(state):
            return {}

        async def branch_b(state):
            return {}

        nodes = make_nodes(a=a, branch_a=branch_a, branch_b=branch_b)
        edges = [
            Edge(START, "a"),
            Edge("a", ["branch_a", "branch_b"]),
            Edge("branch_a", END),
            Edge("branch_b", END),
        ]
        graph = build_graph(nodes, edges)
        loop = EngineLoop(nodes, graph, reducers={})

        ctx = ExecutionContext(session_id="s1", state={})
        events = await drain(loop.run(ctx, "a"))

        assert all(e.branch_index is None for e in events)
        assert all(e.branch_total is None for e in events)
        assert all(e.dispatch_id is None for e in events)

    async def test_on_branch_hooks_fire_once_per_branch_alongside_node_hooks(self):
        node_starts = []
        node_ends = []
        branch_starts = []
        branch_ends = []

        async def process_item(state):
            return {}

        def route(state):
            return [Send("process_item", {"item": i}) for i in range(3)]

        nodes = make_nodes(split=lambda s: {}, process_item=process_item)
        edges = [
            Edge(START, "split"),
            Edge("split", ["process_item"], condition=route),
            Edge("process_item", END),
        ]
        graph = build_graph(nodes, edges)
        hooks = FlowEngineHooks(
            on_node_start=lambda name, state: node_starts.append(name),
            on_node_end=lambda name, state, updates: node_ends.append(name),
            on_branch_start=lambda name, index, total, state: branch_starts.append(
                (name, index, total)
            ),
            on_branch_end=lambda name, index, total, state, updates: branch_ends.append(
                (name, index, total)
            ),
        )
        loop = EngineLoop(nodes, graph, reducers={}, hooks=hooks)

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "split"))

        assert node_starts.count("process_item") == 3
        assert node_ends.count("process_item") == 3
        assert sorted(branch_starts) == [
            ("process_item", 0, 3),
            ("process_item", 1, 3),
            ("process_item", 2, 3),
        ]
        assert sorted(branch_ends) == [
            ("process_item", 0, 3),
            ("process_item", 1, 3),
            ("process_item", 2, 3),
        ]

    async def test_on_branch_hooks_not_fired_for_plain_node_or_static_fan_out(self):
        branch_starts = []

        async def a(state):
            return {}

        nodes = make_nodes(a=a)
        edges = [Edge(START, "a"), Edge("a", END)]
        graph = build_graph(nodes, edges)
        hooks = FlowEngineHooks(
            on_branch_start=lambda name, index, total, state: branch_starts.append(name)
        )
        loop = EngineLoop(nodes, graph, reducers={}, hooks=hooks)

        ctx = ExecutionContext(session_id="s1", state={})
        await drain(loop.run(ctx, "a"))

        assert branch_starts == []
