"""Unit tests for llmfy/flow_engine/checkpointer/in_memory_checkpointer.py."""

from datetime import UTC, datetime, timedelta

from llmfy.flow_engine.checkpointer.base_checkpointer import (
    Checkpoint,
    CheckpointMetadata,
)
from llmfy.flow_engine.checkpointer.in_memory_checkpointer import InMemoryCheckpointer


def make_checkpoint(
    session_id: str,
    checkpoint_id: str,
    step: int,
    node: str = "n",
    run_id: str = "run-1",
    prev_node: str = "__start__",
    updated_fields: list[str] | None = None,
    attempt: int | None = 1,
    dispatch_id: str | None = None,
) -> Checkpoint:
    return Checkpoint(
        metadata=CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            session_id=session_id,
            run_id=run_id,
            created_at=datetime.now(UTC) + timedelta(microseconds=step),
            node=node,
            prev_node=prev_node,
            step=step,
            updated_fields=updated_fields if updated_fields is not None else [],
            attempt=attempt,
            dispatch_id=dispatch_id,
        ),
        state={"step": step},
    )


class TestSaveAndLoad:
    async def test_load_latest_returns_last_saved(self):
        checkpointer = InMemoryCheckpointer()
        await checkpointer.save(make_checkpoint("s1", "c1", 1))
        await checkpointer.save(make_checkpoint("s1", "c2", 2))

        latest = await checkpointer.load("s1")
        assert latest is not None
        assert latest.metadata.checkpoint_id == "c2"
        assert latest.state == {"step": 2}

    async def test_load_specific_checkpoint_id(self):
        checkpointer = InMemoryCheckpointer()
        await checkpointer.save(make_checkpoint("s1", "c1", 1))
        await checkpointer.save(make_checkpoint("s1", "c2", 2))

        found = await checkpointer.load("s1", checkpoint_id="c1")
        assert found is not None
        assert found.metadata.checkpoint_id == "c1"

    async def test_load_wrong_session_for_checkpoint_id_returns_none(self):
        checkpointer = InMemoryCheckpointer()
        await checkpointer.save(make_checkpoint("s1", "c1", 1))

        assert await checkpointer.load("other-session", checkpoint_id="c1") is None

    async def test_load_missing_session_returns_none(self):
        checkpointer = InMemoryCheckpointer()
        assert await checkpointer.load("missing") is None

    async def test_saved_state_is_isolated_from_caller_mutation(self):
        checkpointer = InMemoryCheckpointer()
        cp = make_checkpoint("s1", "c1", 1)
        await checkpointer.save(cp)
        cp.state["step"] = 999

        stored = await checkpointer.load("s1")
        assert stored.state == {"step": 1}  # type: ignore


class TestList:
    async def test_list_returns_newest_first(self):
        checkpointer = InMemoryCheckpointer()
        await checkpointer.save(make_checkpoint("s1", "c1", 1))
        await checkpointer.save(make_checkpoint("s1", "c2", 2))
        await checkpointer.save(make_checkpoint("s1", "c3", 3))

        results = await checkpointer.list("s1")
        assert [c.metadata.checkpoint_id for c in results] == ["c3", "c2", "c1"]

    async def test_list_respects_limit(self):
        checkpointer = InMemoryCheckpointer()
        for i in range(5):
            await checkpointer.save(make_checkpoint("s1", f"c{i}", i))

        results = await checkpointer.list("s1", limit=2)
        assert len(results) == 2

    async def test_list_missing_session_returns_empty(self):
        checkpointer = InMemoryCheckpointer()
        assert await checkpointer.list("missing") == []


class TestDelete:
    async def test_delete_specific_checkpoint(self):
        checkpointer = InMemoryCheckpointer()
        await checkpointer.save(make_checkpoint("s1", "c1", 1))
        await checkpointer.save(make_checkpoint("s1", "c2", 2))

        await checkpointer.delete("s1", checkpoint_id="c1")

        assert await checkpointer.load("s1", checkpoint_id="c1") is None
        assert await checkpointer.load("s1", checkpoint_id="c2") is not None

    async def test_delete_all_for_session(self):
        checkpointer = InMemoryCheckpointer()
        await checkpointer.save(make_checkpoint("s1", "c1", 1))
        await checkpointer.save(make_checkpoint("s1", "c2", 2))

        await checkpointer.delete("s1")

        assert await checkpointer.list("s1") == []

    async def test_delete_does_not_affect_other_sessions(self):
        checkpointer = InMemoryCheckpointer()
        await checkpointer.save(make_checkpoint("s1", "c1", 1))
        await checkpointer.save(make_checkpoint("s2", "c2", 1))

        await checkpointer.delete("s1")

        assert await checkpointer.load("s2") is not None


class TestClearAll:
    async def test_clear_all_removes_every_session(self):
        checkpointer = InMemoryCheckpointer()
        await checkpointer.save(make_checkpoint("s1", "c1", 1))
        await checkpointer.save(make_checkpoint("s2", "c2", 1))

        await checkpointer.clear_all()

        assert await checkpointer.list("s1") == []
        assert await checkpointer.list("s2") == []


class TestRetention:
    async def test_default_is_unbounded(self):
        checkpointer = InMemoryCheckpointer()
        for i in range(10):
            await checkpointer.save(make_checkpoint("s1", f"c{i}", i))

        assert len(await checkpointer.list("s1", limit=100)) == 10

    async def test_max_checkpoints_per_session_drops_oldest(self):
        checkpointer = InMemoryCheckpointer(max_checkpoints_per_session=2)
        for i in range(5):
            await checkpointer.save(make_checkpoint("s1", f"c{i}", i))

        results = await checkpointer.list("s1", limit=100)
        assert [c.metadata.checkpoint_id for c in results] == ["c4", "c3"]

    async def test_dropped_checkpoints_are_unreachable_by_id(self):
        checkpointer = InMemoryCheckpointer(max_checkpoints_per_session=1)
        await checkpointer.save(make_checkpoint("s1", "c0", 0))
        await checkpointer.save(make_checkpoint("s1", "c1", 1))

        assert await checkpointer.load("s1", checkpoint_id="c0") is None
        assert await checkpointer.load("s1", checkpoint_id="c1") is not None

    async def test_retention_is_per_session(self):
        checkpointer = InMemoryCheckpointer(max_checkpoints_per_session=1)
        await checkpointer.save(make_checkpoint("s1", "c1", 1))
        await checkpointer.save(make_checkpoint("s2", "c2", 1))

        assert len(await checkpointer.list("s1", limit=100)) == 1
        assert len(await checkpointer.list("s2", limit=100)) == 1

    async def test_default_ttl_is_none_keeps_old_checkpoints(self):
        checkpointer = InMemoryCheckpointer()
        old = make_checkpoint("s1", "c-old", 0)
        old.metadata.created_at = datetime.now(UTC) - timedelta(hours=1)
        await checkpointer.save(old)

        assert await checkpointer.load("s1", checkpoint_id="c-old") is not None

    async def test_ttl_seconds_drops_expired_checkpoints_on_save(self):
        checkpointer = InMemoryCheckpointer(ttl_seconds=60)
        old = make_checkpoint("s1", "c-old", 0)
        old.metadata.created_at = datetime.now(UTC) - timedelta(hours=1)
        await checkpointer.save(old)

        # Triggers the age check that prunes "c-old".
        await checkpointer.save(make_checkpoint("s1", "c-new", 1))

        assert await checkpointer.load("s1", checkpoint_id="c-old") is None
        assert await checkpointer.load("s1", checkpoint_id="c-new") is not None

    async def test_ttl_seconds_keeps_fresh_checkpoints(self):
        checkpointer = InMemoryCheckpointer(ttl_seconds=3600)
        await checkpointer.save(make_checkpoint("s1", "c1", 0))
        await checkpointer.save(make_checkpoint("s1", "c2", 1))

        assert len(await checkpointer.list("s1", limit=100)) == 2

    async def test_ttl_and_count_cap_combine(self):
        checkpointer = InMemoryCheckpointer(max_checkpoints_per_session=1, ttl_seconds=60)
        old = make_checkpoint("s1", "c-old", 0)
        old.metadata.created_at = datetime.now(UTC) - timedelta(hours=1)
        await checkpointer.save(old)
        await checkpointer.save(make_checkpoint("s1", "c-new", 1))

        results = await checkpointer.list("s1", limit=100)
        assert [c.metadata.checkpoint_id for c in results] == ["c-new"]


class TestGetStats:
    async def test_reflects_sessions_and_checkpoints(self):
        checkpointer = InMemoryCheckpointer()
        await checkpointer.save(make_checkpoint("s1", "c1", 1))
        await checkpointer.save(make_checkpoint("s1", "c2", 2))
        await checkpointer.save(make_checkpoint("s2", "c3", 1))

        stats = checkpointer.get_stats()
        assert stats["total_sessions"] == 2
        assert stats["total_checkpoints"] == 3
        assert stats["checkpoints_per_session"]["s1"] == 2
        assert stats["checkpoints_per_session"]["s2"] == 1
