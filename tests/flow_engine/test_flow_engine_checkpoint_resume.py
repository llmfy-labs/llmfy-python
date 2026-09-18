"""Unit tests for FlowEngine checkpoint save/resume and checkpoint
management methods (get_state, list_checkpoints, get_checkpoint,
delete_checkpoints, reset_session)."""

from dataclasses import dataclass
from typing import Annotated, TypedDict

import pytest

from llmfy.exception.llmfy_exception import (
    CheckpointDeserializationException,
    InvalidSessionIdException,
)
from llmfy.flow_engine.checkpointer.in_memory_checkpointer import InMemoryCheckpointer
from llmfy.flow_engine.flow_engine import FlowEngine
from llmfy.flow_engine.node.node import END, START


def add_reducer(old, new):
    return (old or 0) + new


class AppState(TypedDict):
    count: Annotated[int, add_reducer]


@dataclass
class Payload:
    value: str


class RequiresSerialization(InMemoryCheckpointer):
    requires_serialization = True


class TestCheckpointSaveDuringInvoke:
    async def test_checkpoint_saved_after_each_node(self):
        async def a(state):
            return {"count": 1}

        async def b(state):
            return {"count": 1}

        checkpointer = InMemoryCheckpointer()
        flow = FlowEngine(AppState, checkpointer=checkpointer)
        flow.add_node("a", a)
        flow.add_node("b", b)
        flow.add_edge(START, "a")
        flow.add_edge("a", "b")
        flow.add_edge("b", END)
        flow.build()

        await flow.invoke(session_id="s1")

        checkpoints = await flow.list_checkpoints("s1", limit=100)
        node_names = {c.metadata.node for c in checkpoints}
        assert "a" in node_names
        assert "b" in node_names

    async def test_get_state_returns_latest_state(self):
        async def a(state):
            return {"count": 5}

        checkpointer = InMemoryCheckpointer()
        flow = FlowEngine(AppState, checkpointer=checkpointer)
        flow.add_node("a", a)
        flow.add_edge(START, "a")
        flow.add_edge("a", END)
        flow.build()

        await flow.invoke(session_id="s1")

        state = await flow.get_state("s1")
        assert state == {"count": 5}


class TestResume:
    async def test_resume_continues_after_last_completed_node(self):
        calls = []
        b_should_fail = {"value": True}

        async def a(state):
            calls.append("a")
            return {"count": 1}

        async def b(state):
            calls.append("b")
            if b_should_fail["value"]:
                raise ValueError("simulated crash before checkpointing 'b'")
            return {"count": 1}

        checkpointer = InMemoryCheckpointer()
        flow = FlowEngine(AppState, checkpointer=checkpointer)
        flow.add_node("a", a)
        flow.add_node("b", b)
        flow.add_edge(START, "a")
        flow.add_edge("a", "b")
        flow.add_edge("b", END)
        flow.build()

        # "a" completes and checkpoints; "b" runs but crashes before its own
        # checkpoint is saved — the latest checkpoint on session "s1" is "a".
        from llmfy.exception.llmfy_exception import NodeExecutionException

        with pytest.raises(NodeExecutionException):
            await flow.invoke(session_id="s1")
        assert calls == ["a", "b"]

        # Resume: should continue from "b" (not re-run "a"), now succeeding.
        b_should_fail["value"] = False
        result = await flow.invoke(session_id="s1")
        assert calls == ["a", "b", "b"]
        assert result["count"] == 2

    async def test_apply_state_on_resume_merges_via_reducer(self):
        async def a(state):
            return {"count": 1}

        checkpointer = InMemoryCheckpointer()
        flow = FlowEngine(AppState, checkpointer=checkpointer)
        flow.add_node("a", a)
        flow.add_edge(START, "a")
        flow.add_edge("a", END)
        flow.build()

        await flow.invoke(session_id="s1")  # count == 1, workflow completed

        # Completed workflows resume from START again (no next node after
        # the last completed one that reached END).
        result = await flow.invoke(session_id="s1", apply_state={"count": 10})
        assert result["count"] == 1 + 10 + 1  # prior(1) + applied(10) + node(1)

    async def test_no_session_id_always_starts_fresh(self):
        async def a(state):
            return {"count": 1}

        checkpointer = InMemoryCheckpointer()
        flow = FlowEngine(AppState, checkpointer=checkpointer)
        flow.add_node("a", a)
        flow.add_edge(START, "a")
        flow.add_edge("a", END)
        flow.build()

        result_1 = await flow.invoke()
        result_2 = await flow.invoke()
        assert result_1["count"] == 1
        assert result_2["count"] == 1


class TestPerRunStepBudget:
    """`max_steps` guards one `invoke()`/`stream()` call, not a
    `session_id`'s cumulative lifetime. A session reused across many calls
    (e.g. a multi-turn chat loop, each turn its own `invoke()`) must not
    accumulate a shared step count that eventually trips the limit on an
    otherwise-fine call."""

    async def test_step_budget_resets_each_call_not_cumulative_across_session(self):
        async def a(state):
            return {"count": 1}

        checkpointer = InMemoryCheckpointer()
        # max_steps=1: each call executes exactly one node ("a"), landing
        # exactly at the limit. Before the fix, `step` was restored from
        # the checkpoint (already 1 after the first call), so the second
        # call would start pre-tripped and raise on its first node.
        flow = FlowEngine(AppState, checkpointer=checkpointer, max_steps=1)
        flow.add_node("a", a)
        flow.add_edge(START, "a")
        flow.add_edge("a", END)
        flow.build()

        for expected_count in (1, 2, 3):
            result = await flow.invoke(session_id="s1")
            assert result["count"] == expected_count

    async def test_step_budget_resets_on_resume_after_a_crash(self):
        calls = []
        b_should_fail = {"value": True}

        async def a(state):
            calls.append("a")
            return {"count": 1}

        async def b(state):
            calls.append("b")
            if b_should_fail["value"]:
                raise ValueError("simulated crash before checkpointing 'b'")
            return {"count": 1}

        checkpointer = InMemoryCheckpointer()
        # max_steps=2: exactly enough for "a" then "b" in one call. Before
        # the fix, resuming after "a" restored step=1, so the resumed call
        # would need 2 more steps to finish "b" — 3 total — and trip a
        # limit sized for a single run.
        flow = FlowEngine(AppState, checkpointer=checkpointer, max_steps=2)
        flow.add_node("a", a)
        flow.add_node("b", b)
        flow.add_edge(START, "a")
        flow.add_edge("a", "b")
        flow.add_edge("b", END)
        flow.build()

        from llmfy.exception.llmfy_exception import NodeExecutionException

        with pytest.raises(NodeExecutionException):
            await flow.invoke(session_id="s1")

        b_should_fail["value"] = False
        result = await flow.invoke(session_id="s1")
        assert calls == ["a", "b", "b"]
        assert result["count"] == 2


class TestRunIdGroupsOneCallsCheckpoints:
    """`run_id` identifies one `invoke()`/`stream()` call — every
    checkpoint saved during that call shares it, distinct from
    `session_id` which spans a session's whole lifetime across calls."""

    async def test_checkpoints_within_one_call_share_run_id(self):
        async def a(state):
            return {"count": 1}

        async def b(state):
            return {"count": 1}

        checkpointer = InMemoryCheckpointer()
        flow = FlowEngine(AppState, checkpointer=checkpointer)
        flow.add_node("a", a)
        flow.add_node("b", b)
        flow.add_edge(START, "a")
        flow.add_edge("a", "b")
        flow.add_edge("b", END)
        flow.build()

        await flow.invoke(session_id="s1")

        checkpoints = await flow.list_checkpoints("s1", limit=100)
        assert len(checkpoints) == 2  # one per node: "a" and "b"
        run_ids = {c.metadata.run_id for c in checkpoints}
        assert len(run_ids) == 1
        assert next(iter(run_ids))  # non-empty

    async def test_each_call_gets_its_own_run_id(self):
        async def a(state):
            return {"count": 1}

        checkpointer = InMemoryCheckpointer()
        flow = FlowEngine(AppState, checkpointer=checkpointer)
        flow.add_node("a", a)
        flow.add_edge(START, "a")
        flow.add_edge("a", END)
        flow.build()

        await flow.invoke(session_id="s1")
        first = await flow.get_checkpoint("s1")

        await flow.invoke(session_id="s1")
        second = await flow.get_checkpoint("s1")

        assert first.metadata.run_id != second.metadata.run_id  # type: ignore[union-attr]

    async def test_resumed_call_gets_a_new_run_id_not_the_interrupted_ones(self):
        b_should_fail = {"value": True}

        async def a(state):
            return {"count": 1}

        async def b(state):
            if b_should_fail["value"]:
                raise ValueError("simulated crash before checkpointing 'b'")
            return {"count": 1}

        checkpointer = InMemoryCheckpointer()
        flow = FlowEngine(AppState, checkpointer=checkpointer)
        flow.add_node("a", a)
        flow.add_node("b", b)
        flow.add_edge(START, "a")
        flow.add_edge("a", "b")
        flow.add_edge("b", END)
        flow.build()

        from llmfy.exception.llmfy_exception import NodeExecutionException

        with pytest.raises(NodeExecutionException):
            await flow.invoke(session_id="s1")
        interrupted_run_id = (await flow.get_checkpoint("s1")).metadata.run_id  # type: ignore[union-attr]

        b_should_fail["value"] = False
        await flow.invoke(session_id="s1")
        resumed_run_id = (await flow.get_checkpoint("s1")).metadata.run_id  # type: ignore[union-attr]

        assert resumed_run_id != interrupted_run_id


class TestPrevNode:
    """`prev_node` records whichever node's completion led to `node`
    running — `START` for the first node of a run, fresh or resumed."""

    async def test_first_node_of_a_fresh_run_has_start_as_prev(self):
        async def a(state):
            return {"count": 1}

        checkpointer = InMemoryCheckpointer()
        flow = FlowEngine(AppState, checkpointer=checkpointer)
        flow.add_node("a", a)
        flow.add_edge(START, "a")
        flow.add_edge("a", END)
        flow.build()

        await flow.invoke(session_id="s1")

        checkpoint = await flow.get_checkpoint("s1")
        assert checkpoint.metadata.node == "a"  # type: ignore[union-attr]
        assert checkpoint.metadata.prev_node == START  # type: ignore[union-attr]

    async def test_later_nodes_record_their_actual_predecessor(self):
        async def a(state):
            return {"count": 1}

        async def b(state):
            return {"count": 1}

        checkpointer = InMemoryCheckpointer()
        flow = FlowEngine(AppState, checkpointer=checkpointer)
        flow.add_node("a", a)
        flow.add_node("b", b)
        flow.add_edge(START, "a")
        flow.add_edge("a", "b")
        flow.add_edge("b", END)
        flow.build()

        await flow.invoke(session_id="s1")

        checkpoints = await flow.list_checkpoints("s1", limit=100)
        by_node = {c.metadata.node: c.metadata.prev_node for c in checkpoints}
        assert by_node["a"] == START
        assert by_node["b"] == "a"

    async def test_first_node_of_a_resumed_run_also_has_start_as_prev(self):
        # A completed workflow resumes from START again (no next node
        # after the last completed one that reached END) — matches
        # TestResume.test_apply_state_on_resume_merges_via_reducer.
        async def a(state):
            return {"count": 1}

        checkpointer = InMemoryCheckpointer()
        flow = FlowEngine(AppState, checkpointer=checkpointer)
        flow.add_node("a", a)
        flow.add_edge(START, "a")
        flow.add_edge("a", END)
        flow.build()

        await flow.invoke(session_id="s1")
        await flow.invoke(session_id="s1")  # second run, resumed

        checkpoint = await flow.get_checkpoint("s1")
        assert checkpoint.metadata.prev_node == START  # type: ignore[union-attr]

    async def test_first_node_of_a_run_resumed_mid_graph_still_has_start_as_prev(
        self,
    ):
        # A resumed run picks up the graph *position* from the last
        # checkpoint (continues at "b", not "a" — see
        # TestResume.test_resume_continues_after_last_completed_node), but
        # from this run's own perspective "b" is the first node it ran:
        # prev_node is START, not "a", matching run_id/step's per-run reset
        # (TestPerRunStepBudget) rather than "b"'s actual graph predecessor.
        b_should_fail = {"value": True}

        async def a(state):
            return {"count": 1}

        async def b(state):
            if b_should_fail["value"]:
                raise ValueError("simulated crash before checkpointing 'b'")
            return {"count": 1}

        checkpointer = InMemoryCheckpointer()
        flow = FlowEngine(AppState, checkpointer=checkpointer)
        flow.add_node("a", a)
        flow.add_node("b", b)
        flow.add_edge(START, "a")
        flow.add_edge("a", "b")
        flow.add_edge("b", END)
        flow.build()

        from llmfy.exception.llmfy_exception import NodeExecutionException

        with pytest.raises(NodeExecutionException):
            await flow.invoke(session_id="s1")

        b_should_fail["value"] = False
        await flow.invoke(session_id="s1")

        checkpoint = await flow.get_checkpoint("s1")
        assert checkpoint.metadata.node == "b"  # type: ignore[union-attr]
        assert checkpoint.metadata.prev_node == START  # type: ignore[union-attr]


class TestCheckpointManagementMethods:
    async def test_methods_require_checkpointer(self):
        from llmfy.exception.llmfy_exception import GraphValidationException

        flow = FlowEngine(AppState)
        flow.add_node("a", lambda s: {"count": 1})
        flow.add_edge(START, "a")
        flow.add_edge("a", END)
        flow.build()

        with pytest.raises(GraphValidationException, match="No checkpointer"):
            await flow.get_state("s1")
        with pytest.raises(GraphValidationException, match="No checkpointer"):
            await flow.list_checkpoints("s1")
        with pytest.raises(GraphValidationException, match="No checkpointer"):
            await flow.delete_checkpoints("s1")
        with pytest.raises(GraphValidationException, match="No checkpointer"):
            await flow.reset_session("s1")

    async def test_delete_checkpoints_removes_session_state(self):
        async def a(state):
            return {"count": 1}

        checkpointer = InMemoryCheckpointer()
        flow = FlowEngine(AppState, checkpointer=checkpointer)
        flow.add_node("a", a)
        flow.add_edge(START, "a")
        flow.add_edge("a", END)
        flow.build()

        await flow.invoke(session_id="s1")
        await flow.delete_checkpoints("s1")

        assert await flow.get_state("s1") is None

    async def test_reset_session_allows_fresh_restart(self):
        async def a(state):
            return {"count": 1}

        checkpointer = InMemoryCheckpointer()
        flow = FlowEngine(AppState, checkpointer=checkpointer)
        flow.add_node("a", a)
        flow.add_edge(START, "a")
        flow.add_edge("a", END)
        flow.build()

        await flow.invoke(session_id="s1")
        await flow.reset_session("s1")

        result = await flow.invoke(session_id="s1")
        assert result["count"] == 1


class TestTypeRegistryIntegration:
    async def test_registered_type_round_trips_through_external_checkpointer(self):
        async def a(state):
            return {"payload": Payload(value="x")}

        class AppStateWithPayload(TypedDict):
            payload: Payload

        checkpointer = RequiresSerialization()
        flow = FlowEngine(
            AppStateWithPayload, checkpointer=checkpointer, types=[Payload]
        )
        flow.add_node("a", a)
        flow.add_edge(START, "a")
        flow.add_edge("a", END)
        flow.build()

        await flow.invoke(session_id="s1")

        state = await flow.get_state("s1")
        assert state["payload"] == Payload(value="x")  # type: ignore

    async def test_unregistered_type_raises_on_external_checkpointer(self):
        async def a(state):
            return {"payload": Payload(value="x")}

        class AppStateWithPayload(TypedDict):
            payload: Payload

        checkpointer = RequiresSerialization()
        flow = FlowEngine(
            AppStateWithPayload, checkpointer=checkpointer
        )  # not registered
        flow.add_node("a", a)
        flow.add_edge(START, "a")
        flow.add_edge("a", END)
        flow.build()

        with pytest.raises(CheckpointDeserializationException):
            await flow.invoke(session_id="s1")


class TestSessionIdValidation:
    def _build_flow(self) -> FlowEngine:
        async def a(state):
            return {"count": 1}

        flow = FlowEngine(AppState, checkpointer=InMemoryCheckpointer())
        flow.add_node("a", a)
        flow.add_edge(START, "a")
        flow.add_edge("a", END)
        flow.build()
        return flow

    async def test_auto_generated_uuid_session_id_is_accepted(self):
        flow = self._build_flow()
        await flow.invoke()  # no session_id -> engine generates a uuid4

    async def test_valid_custom_session_id_is_accepted(self):
        flow = self._build_flow()
        await flow.invoke(session_id="user-42:chat.session-1")

    async def test_empty_string_session_id_falls_back_to_auto_generated(self):
        # Pre-existing `session_id or uuid4()` behavior: "" is falsy, so it
        # is replaced before ever reaching the regex check, same as `None`.
        flow = self._build_flow()
        await flow.invoke(session_id="")

    @pytest.mark.parametrize(
        "bad_session_id",
        [
            "a/b",
            "a b",
            "a" * 256,
            "session?id=1",
            "../../etc/passwd",
        ],
    )
    async def test_invalid_session_id_is_rejected(self, bad_session_id):
        flow = self._build_flow()
        with pytest.raises(InvalidSessionIdException) as exc_info:
            await flow.invoke(session_id=bad_session_id)
        assert exc_info.value.session_id == bad_session_id

    async def test_invalid_session_id_is_rejected_for_stream(self):
        flow = self._build_flow()
        with pytest.raises(InvalidSessionIdException):
            async for _ in flow.stream(session_id="bad/id"):
                pass

    async def test_register_type_after_construction(self):
        async def a(state):
            return {"payload": Payload(value="y")}

        class AppStateWithPayload(TypedDict):
            payload: Payload

        checkpointer = RequiresSerialization()
        flow = FlowEngine(AppStateWithPayload, checkpointer=checkpointer)
        flow.register_type(Payload)
        flow.add_node("a", a)
        flow.add_edge(START, "a")
        flow.add_edge("a", END)
        flow.build()

        await flow.invoke(session_id="s1")
        state = await flow.get_state("s1")
        assert state["payload"] == Payload(value="y") # type: ignore
