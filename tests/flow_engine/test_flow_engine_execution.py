"""Unit tests for FlowEngine.invoke() and related public execution API."""

from typing import Annotated, TypedDict

import pytest

from llmfy.exception.llmfy_exception import StepLimitExceededException
from llmfy.flow_engine.execution.hooks import FlowEngineHooks
from llmfy.flow_engine.execution.policy import RetryPolicy
from llmfy.flow_engine.execution.send import Send
from llmfy.flow_engine.flow_engine import FlowEngine
from llmfy.flow_engine.node.node import END, START


def union_reducer(old, new):
    return (old or set()) | set(new)


class AppState(TypedDict):
    count: int
    visited: Annotated[set, union_reducer]


class TestInvokeLinear:
    async def test_returns_final_state(self):
        async def step_a(state):
            return {"count": state.get("count", 0) + 1}

        async def step_b(state):
            return {"count": state.get("count", 0) + 1}

        flow = FlowEngine(AppState)
        flow.add_node("a", step_a)
        flow.add_node("b", step_b)
        flow.add_edge(START, "a")
        flow.add_edge("a", "b")
        flow.add_edge("b", END)
        flow.build()

        result = await flow.invoke()
        assert result["count"] == 2

    async def test_apply_state_seeds_initial_state(self):
        async def step_a(state):
            return {"count": state["count"] + 1}

        flow = FlowEngine(AppState)
        flow.add_node("a", step_a)
        flow.add_edge(START, "a")
        flow.add_edge("a", END)
        flow.build()

        result = await flow.invoke(apply_state={"count": 10})
        assert result["count"] == 11


class TestInvokeConditional:
    async def test_condition_routes_to_declared_target(self):
        async def looper(state):
            return {"count": state.get("count", 0) + 1}

        def route(state):
            return "looper" if state["count"] < 3 else END

        flow = FlowEngine(AppState)
        flow.add_node("looper", looper)
        flow.add_edge(START, "looper")
        flow.add_conditional_edges("looper", route, ["looper", END])
        flow.build()

        result = await flow.invoke()
        assert result["count"] == 3

    async def test_condition_routes_via_target_map_dict(self):
        """LangGraph-style routing: condition_func returns a label, and the
        dict maps that label to the actual target node."""

        async def looper(state):
            return {"count": state.get("count", 0) + 1}

        def route(state):
            return "continue" if state["count"] < 3 else "stop"

        flow = FlowEngine(AppState)
        flow.add_node("looper", looper)
        flow.add_edge(START, "looper")
        flow.add_conditional_edges(
            "looper", route, {"continue": "looper", "stop": END}
        )
        flow.build()

        result = await flow.invoke()
        assert result["count"] == 3

    async def test_loop_back_edge_to_entry_node_keeps_looping(self):
        """Regression: an agent-style graph where `main` is both the START
        target and the target of a regular loop-back edge from `tools`.
        `main` previously mis-detected as a join node (START + `tools` both
        counted as predecessors), so once `tools` looped back, execution
        silently stopped instead of re-entering `main` — see
        test_graph_builder.TestStartEdge.test_entry_node_looped_back_to_is_not_a_join
        for the underlying graph-structure assertion."""

        async def main(state):
            return {"count": state.get("count", 0) + 1}

        async def tools(state):
            return {}

        def route(state):
            return "tools" if state["count"] < 2 else END

        flow = FlowEngine(AppState)
        flow.add_node("main", main)
        flow.add_node("tools", tools)
        flow.add_edge(START, "main")
        flow.add_edge("tools", "main")
        flow.add_conditional_edges("main", route, ["tools", END])
        flow.build()

        result = await flow.invoke()
        assert result["count"] == 2

    async def test_target_map_unknown_key_raises(self):
        from llmfy.exception.llmfy_exception import GraphValidationException

        async def looper(state):
            return {}

        def route(state):
            return "not-a-key"

        flow = FlowEngine(AppState)
        flow.add_node("looper", looper)
        flow.add_edge(START, "looper")
        flow.add_conditional_edges("looper", route, {"continue": "looper", "stop": END})
        flow.build()

        with pytest.raises(GraphValidationException, match="not a key in its target map"):
            await flow.invoke()


class TestInvokeFanOutFanIn:
    async def test_fan_out_via_public_api(self):
        async def fetch(state):
            return {}

        async def branch_a(state):
            return {"visited": {"a"}}

        async def branch_b(state):
            return {"visited": {"b"}}

        async def combine(state):
            return {}

        flow = FlowEngine(AppState)
        flow.add_node("fetch", fetch)
        flow.add_node("branch_a", branch_a)
        flow.add_node("branch_b", branch_b)
        flow.add_node("combine", combine)
        flow.add_edge(START, "fetch")
        flow.add_edge("fetch", ["branch_a", "branch_b"])
        flow.add_edge("branch_a", "combine")
        flow.add_edge("branch_b", "combine")
        flow.add_edge("combine", END)
        flow.build()

        result = await flow.invoke()
        assert result["visited"] == {"a", "b"}


class TestDynamicFanOut:
    async def test_send_based_map_via_public_api(self):
        """add_conditional_edges' public signature needs no change for
        Send — only the routing function's return type changes (a plain
        node-name string vs. a runtime-determined list[Send])."""

        async def plan(state):
            return {}

        async def process_item(state):
            return {"visited": {state["item"]}}

        def route_map(state):
            return [Send(node="process_item", state={"item": i}) for i in range(4)]

        flow = FlowEngine(AppState)
        flow.add_node("plan", plan)
        flow.add_node("process_item", process_item)
        flow.add_edge(START, "plan")
        flow.add_conditional_edges("plan", route_map, ["process_item"])
        flow.add_edge("process_item", END)
        flow.build()

        result = await flow.invoke()
        assert result["visited"] == {0, 1, 2, 3}

    async def test_stream_events_carry_branch_correlation_ids(self):
        async def plan(state):
            return {}

        async def process_item(state):
            return {"visited": {state["item"]}}

        def route_map(state):
            return [Send(node="process_item", state={"item": i}) for i in range(3)]

        flow = FlowEngine(AppState)
        flow.add_node("plan", plan)
        flow.add_node("process_item", process_item)
        flow.add_edge(START, "plan")
        flow.add_conditional_edges("plan", route_map, ["process_item"])
        flow.add_edge("process_item", END)
        flow.build()

        responses = [r async for r in flow.stream()]
        branch_events = [
            r for r in responses if r.node == "process_item" and r.type == "result"
        ]

        assert len(branch_events) == 3
        indices = [r.branch_index for r in branch_events if r.branch_index is not None]
        assert len(indices) == 3
        assert sorted(indices) == [0, 1, 2]
        assert all(r.branch_total == 3 for r in branch_events)
        assert len({r.dispatch_id for r in branch_events}) == 1
        assert branch_events[0].dispatch_id is not None


class TestMaxSteps:
    async def test_instance_default_max_steps_applies(self):
        async def looper(state):
            return {}

        flow = FlowEngine(AppState, max_steps=3)
        flow.add_node("looper", looper)
        flow.add_edge(START, "looper")
        flow.add_conditional_edges("looper", lambda s: "looper", ["looper", END])
        flow.build()

        with pytest.raises(StepLimitExceededException) as exc_info:
            await flow.invoke()
        assert exc_info.value.max_steps == 3

    async def test_per_call_max_steps_overrides_instance_default(self):
        async def looper(state):
            return {}

        flow = FlowEngine(AppState, max_steps=100)
        flow.add_node("looper", looper)
        flow.add_edge(START, "looper")
        flow.add_conditional_edges("looper", lambda s: "looper", ["looper", END])
        flow.build()

        with pytest.raises(StepLimitExceededException) as exc_info:
            await flow.invoke(max_steps=2)
        assert exc_info.value.max_steps == 2


class TestRetryAndHooksViaPublicApi:
    async def test_add_node_retry_param_retries_transient_failure(self):
        attempts = {"count": 0}

        async def flaky(state):
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise ValueError("transient")
            return {"count": 1}

        flow = FlowEngine(AppState)
        flow.add_node("flaky", flaky, retry=RetryPolicy(max_attempts=2, retry_on=(ValueError,)))
        flow.add_edge(START, "flaky")
        flow.add_edge("flaky", END)
        flow.build()

        result = await flow.invoke()
        assert result["count"] == 1
        assert attempts["count"] == 2

    async def test_hooks_fire_during_invoke(self):
        events = []

        async def a(state):
            return {"count": 1}

        flow = FlowEngine(
            AppState,
            hooks=FlowEngineHooks(
                on_node_start=lambda name, state: events.append(("start", name)),
                on_node_end=lambda name, state, updates: events.append(("end", name)),
            ),
        )
        flow.add_node("a", a)
        flow.add_edge(START, "a")
        flow.add_edge("a", END)
        flow.build()

        await flow.invoke()
        assert events == [("start", "a"), ("end", "a")]


class TestConcurrentInvokeOnOneInstance:
    async def test_two_sessions_do_not_corrupt_each_other(self):
        import asyncio

        async def work(state):
            await asyncio.sleep(0.01)
            return {"count": state.get("seed", 0)}

        flow = FlowEngine(AppState)
        flow.add_node("work", work)
        flow.add_edge(START, "work")
        flow.add_edge("work", END)
        flow.build()

        result_a, result_b = await asyncio.gather(
            flow.invoke(apply_state={"seed": 1}, session_id="a"),
            flow.invoke(apply_state={"seed": 2}, session_id="b"),
        )

        assert result_a["count"] == 1
        assert result_b["count"] == 2
