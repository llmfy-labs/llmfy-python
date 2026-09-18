"""Unit tests for llmfy/flow_engine/checkpointer/sql_checkpointer.py.

Uses a real, file-backed SQLite database (via the sync driver, which is
built into SQLAlchemy — no extra dependency needed) rather than mocking the
DB layer, since sqlite is self-contained and fast. `aiosqlite` isn't in the
dev dependency group, so these tests exercise the sync code path
(`sqlite:///...`), which is exactly as real a test of the SQL logic as the
async path — both funnel through the same `CheckpointModel`/statements.
"""

from datetime import UTC, datetime, timedelta

import pytest
from cryptography.fernet import Fernet, InvalidToken

from llmfy.exception.llmfy_exception import CheckpointPayloadTooLargeException
from llmfy.flow_engine.checkpointer.base_checkpointer import (
    Checkpoint,
    CheckpointMetadata,
)
from llmfy.flow_engine.checkpointer.sql_checkpointer import SQLCheckpointer


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
            created_at=datetime.now(UTC),
            node=node,
            prev_node=prev_node,
            step=step,
            updated_fields=updated_fields if updated_fields is not None else [],
            attempt=attempt,
            dispatch_id=dispatch_id,
        ),
        state={"step": step, "nested": {"a": [1, 2, 3]}},
    )


@pytest.fixture
async def checkpointer(tmp_path):
    db_path = tmp_path / "checkpoints.db"
    cp = SQLCheckpointer(f"sqlite:///{db_path}")
    yield cp
    await cp.close()


class TestSaveAndLoad:
    async def test_load_latest_returns_last_saved(self, checkpointer):
        await checkpointer.save(make_checkpoint("s1", "c1", 1))
        await checkpointer.save(make_checkpoint("s1", "c2", 2))

        latest = await checkpointer.load("s1")
        assert latest is not None
        assert latest.metadata.checkpoint_id == "c2"
        assert latest.state == {"step": 2, "nested": {"a": [1, 2, 3]}}

    async def test_load_specific_checkpoint_id(self, checkpointer):
        await checkpointer.save(make_checkpoint("s1", "c1", 1))
        await checkpointer.save(make_checkpoint("s1", "c2", 2))

        found = await checkpointer.load("s1", checkpoint_id="c1")
        assert found.metadata.checkpoint_id == "c1"

    async def test_load_missing_returns_none(self, checkpointer):
        assert await checkpointer.load("missing") is None

    async def test_state_round_trips_through_binary_column(self, checkpointer):
        cp = make_checkpoint("s1", "c1", 1)
        await checkpointer.save(cp)

        loaded = await checkpointer.load("s1", checkpoint_id="c1")
        assert loaded.state == cp.state


class TestList:
    async def test_list_returns_newest_first_and_respects_limit(self, checkpointer):
        for i in range(5):
            await checkpointer.save(make_checkpoint("s1", f"c{i}", i))

        results = await checkpointer.list("s1", limit=2)
        assert len(results) == 2
        assert results[0].metadata.checkpoint_id == "c4"


class TestDelete:
    async def test_delete_specific_checkpoint(self, checkpointer):
        await checkpointer.save(make_checkpoint("s1", "c1", 1))
        await checkpointer.save(make_checkpoint("s1", "c2", 2))

        await checkpointer.delete("s1", checkpoint_id="c1")

        assert await checkpointer.load("s1", checkpoint_id="c1") is None
        assert await checkpointer.load("s1", checkpoint_id="c2") is not None

    async def test_delete_all_for_session(self, checkpointer):
        await checkpointer.save(make_checkpoint("s1", "c1", 1))
        await checkpointer.save(make_checkpoint("s1", "c2", 2))

        await checkpointer.delete("s1")

        assert await checkpointer.list("s1") == []


class TestClearAll:
    async def test_clear_all_removes_every_session(self, checkpointer):
        await checkpointer.save(make_checkpoint("s1", "c1", 1))
        await checkpointer.save(make_checkpoint("s2", "c2", 1))

        await checkpointer.clear_all()

        assert await checkpointer.list("s1") == []
        assert await checkpointer.list("s2") == []


class TestCodec:
    async def test_compress_disabled_still_round_trips(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        cp = SQLCheckpointer(f"sqlite:///{db_path}", compress=False)
        try:
            await cp.save(make_checkpoint("s1", "c1", 1))
            loaded = await cp.load("s1", checkpoint_id="c1")
            assert loaded.state == {"step": 1, "nested": {"a": [1, 2, 3]}}
        finally:
            await cp.close()

    async def test_encryption_round_trips_and_hides_plaintext(self, tmp_path):
        import sqlite3

        db_path = tmp_path / "checkpoints.db"
        key = Fernet.generate_key()
        cp = SQLCheckpointer(f"sqlite:///{db_path}", encryption_key=key)
        try:
            checkpoint = make_checkpoint("s1", "c1", 1)
            await cp.save(checkpoint)

            loaded = await cp.load("s1", checkpoint_id="c1")
            assert loaded.state == checkpoint.state

            # Read the raw row through a plain sqlite3 connection (bypassing
            # the codec entirely) — the stored bytes must not contain the
            # plaintext JSON key that a compressed-only encoding would.
            conn = sqlite3.connect(str(db_path))
            raw = conn.execute(
                "SELECT state FROM llmfy_checkpoint WHERE checkpoint_id = ?", ("c1",)
            ).fetchone()[0]
            conn.close()
            assert b"nested" not in raw
        finally:
            await cp.close()

    async def test_wrong_encryption_key_fails_to_decode(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        cp_write = SQLCheckpointer(f"sqlite:///{db_path}", encryption_key=Fernet.generate_key())
        cp_read = SQLCheckpointer(f"sqlite:///{db_path}", encryption_key=Fernet.generate_key())
        try:
            await cp_write.save(make_checkpoint("s1", "c1", 1))
            with pytest.raises(InvalidToken):
                await cp_read.load("s1", checkpoint_id="c1")
        finally:
            await cp_write.close()
            await cp_read.close()

    async def test_oversized_state_is_rejected_before_writing(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        cp = SQLCheckpointer(f"sqlite:///{db_path}", max_state_bytes=10)
        try:
            with pytest.raises(CheckpointPayloadTooLargeException):
                await cp.save(make_checkpoint("s1", "c1", 1))
            assert await cp.load("s1") is None
        finally:
            await cp.close()


class TestRetention:
    async def test_default_is_unbounded(self, checkpointer):
        for i in range(5):
            await checkpointer.save(make_checkpoint("s1", f"c{i}", i))

        assert len(await checkpointer.list("s1", limit=100)) == 5

    async def test_max_checkpoints_per_session_drops_oldest(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        cp = SQLCheckpointer(f"sqlite:///{db_path}", max_checkpoints_per_session=2)
        try:
            for i in range(5):
                await cp.save(make_checkpoint("s1", f"c{i}", i))

            results = await cp.list("s1", limit=100)
            assert {c.metadata.checkpoint_id for c in results} == {"c3", "c4"}
        finally:
            await cp.close()

    async def test_ttl_seconds_drops_expired_checkpoints_on_save(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        cp = SQLCheckpointer(f"sqlite:///{db_path}", ttl_seconds=60)
        try:
            old = make_checkpoint("s1", "c-old", 0)
            old.metadata.created_at = datetime.now(UTC) - timedelta(hours=1)
            await cp.save(old)

            # Triggers the age-based cleanup that prunes "c-old".
            await cp.save(make_checkpoint("s1", "c-new", 1))

            assert await cp.load("s1", checkpoint_id="c-old") is None
            assert await cp.load("s1", checkpoint_id="c-new") is not None
        finally:
            await cp.close()
