"""Unit tests for llmfy/flow_engine/checkpointer/redis_checkpointer.py.

Never hits the network — `redis.asyncio.from_url` is monkeypatched to
return an in-memory `FakeRedis` double implementing just the subset of
commands `RedisCheckpointer` calls (set/expire/zadd/zrevrange/zrange/get/
delete/zrem/scan/close), matching this repo's existing convention of
mocking only the network-call boundary while exercising real logic
otherwise (see tests/conftest.py).
"""

import base64
import fnmatch
import json
from datetime import UTC, datetime

import pytest
from cryptography.fernet import Fernet, InvalidToken

from llmfy.exception.llmfy_exception import CheckpointPayloadTooLargeException
from llmfy.flow_engine.checkpointer import (
    redis_checkpointer as redis_checkpointer_module,
)
from llmfy.flow_engine.checkpointer.base_checkpointer import (
    Checkpoint,
    CheckpointMetadata,
)
from llmfy.flow_engine.checkpointer.redis_checkpointer import RedisCheckpointer


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
        state={"step": step},
    )


class FakeRedis:
    """Minimal in-memory stand-in for `redis.asyncio.Redis`."""

    def __init__(self):
        self.store: dict[str, str] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.ttls: dict[str, int] = {}
        self.closed = False

    async def set(self, key, value):
        self.store[key] = value

    async def expire(self, key, ttl):
        self.ttls[key] = ttl

    async def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(mapping)

    async def zrevrange(self, key, start, end):
        members = sorted(
            self.zsets.get(key, {}).items(), key=lambda kv: kv[1], reverse=True
        )
        names = [m for m, _ in members]
        return names[start:] if end == -1 else names[start : end + 1]

    async def zrange(self, key, start, end):
        members = sorted(self.zsets.get(key, {}).items(), key=lambda kv: kv[1])
        names = [m for m, _ in members]
        return names[start:] if end == -1 else names[start : end + 1]

    async def get(self, key):
        return self.store.get(key)

    async def delete(self, *keys):
        for k in keys:
            self.store.pop(k, None)
            self.zsets.pop(k, None)
            self.ttls.pop(k, None)

    async def zrem(self, key, *members):
        for member in members:
            self.zsets.get(key, {}).pop(member, None)

    async def scan(self, cursor, match=None, count=100):
        pattern = match or "*"
        all_keys = set(self.store.keys()) | set(self.zsets.keys())
        matched = [k for k in all_keys if fnmatch.fnmatch(k, pattern)]
        return 0, matched

    async def close(self):
        self.closed = True


@pytest.fixture
def fake_client():
    return FakeRedis()


@pytest.fixture
def checkpointer(monkeypatch, fake_client):
    async def fake_from_url(*args, **kwargs):
        return fake_client

    monkeypatch.setattr(redis_checkpointer_module.redis, "from_url", fake_from_url)  # type: ignore
    return RedisCheckpointer(redis_url="redis://fake/0")


def make_checkpointer(monkeypatch, fake_client, **kwargs) -> RedisCheckpointer:
    async def fake_from_url(*args, **kw):
        return fake_client

    monkeypatch.setattr(redis_checkpointer_module.redis, "from_url", fake_from_url)  # type: ignore
    return RedisCheckpointer(redis_url="redis://fake/0", **kwargs)


class TestSaveAndLoad:
    async def test_load_latest_returns_last_saved(self, checkpointer):
        await checkpointer.save(make_checkpoint("s1", "c1", 1))
        await checkpointer.save(make_checkpoint("s1", "c2", 2))

        latest = await checkpointer.load("s1")
        assert latest.metadata.checkpoint_id == "c2"
        assert latest.state == {"step": 2}

    async def test_load_specific_checkpoint_id(self, checkpointer):
        await checkpointer.save(make_checkpoint("s1", "c1", 1))
        await checkpointer.save(make_checkpoint("s1", "c2", 2))

        found = await checkpointer.load("s1", checkpoint_id="c1")
        assert found.metadata.checkpoint_id == "c1"

    async def test_load_missing_session_returns_none(self, checkpointer):
        assert await checkpointer.load("missing") is None

    async def test_ttl_sets_expiry_when_configured(self, monkeypatch, fake_client):
        async def fake_from_url(*args, **kwargs):
            return fake_client

        monkeypatch.setattr(redis_checkpointer_module.redis, "from_url", fake_from_url)  # type: ignore
        checkpointer = RedisCheckpointer(redis_url="redis://fake/0", ttl=60)

        await checkpointer.save(make_checkpoint("s1", "c1", 1))

        assert any(ttl == 60 for ttl in fake_client.ttls.values())

    async def test_no_ttl_by_default(self, checkpointer, fake_client):
        await checkpointer.save(make_checkpoint("s1", "c1", 1))
        assert fake_client.ttls == {}


class TestList:
    async def test_list_returns_newest_first(self, checkpointer):
        await checkpointer.save(make_checkpoint("s1", "c1", 1))
        await checkpointer.save(make_checkpoint("s1", "c2", 2))
        await checkpointer.save(make_checkpoint("s1", "c3", 3))

        results = await checkpointer.list("s1")
        assert [c.metadata.checkpoint_id for c in results] == ["c3", "c2", "c1"]

    async def test_list_respects_limit(self, checkpointer):
        for i in range(5):
            await checkpointer.save(make_checkpoint("s1", f"c{i}", i))

        results = await checkpointer.list("s1", limit=2)
        assert len(results) == 2


class TestListByRun:
    async def test_returns_only_checkpoints_from_that_run(self, checkpointer):
        await checkpointer.save(make_checkpoint("s1", "c1", 1, run_id="run-a"))
        await checkpointer.save(make_checkpoint("s1", "c2", 2, run_id="run-a"))
        await checkpointer.save(make_checkpoint("s1", "c3", 1, run_id="run-b"))

        results = await checkpointer.list_by_run("s1", "run-a")
        assert [c.metadata.checkpoint_id for c in results] == ["c2", "c1"]

    async def test_newest_first(self, checkpointer):
        await checkpointer.save(make_checkpoint("s1", "c1", 1, run_id="run-a"))
        await checkpointer.save(make_checkpoint("s1", "c2", 2, run_id="run-a"))
        await checkpointer.save(make_checkpoint("s1", "c3", 3, run_id="run-a"))

        results = await checkpointer.list_by_run("s1", "run-a")
        assert [c.metadata.checkpoint_id for c in results] == ["c3", "c2", "c1"]

    async def test_respects_limit(self, checkpointer):
        for i in range(5):
            await checkpointer.save(make_checkpoint("s1", f"c{i}", i, run_id="run-a"))

        results = await checkpointer.list_by_run("s1", "run-a", limit=2)
        assert len(results) == 2

    async def test_unknown_run_id_returns_empty(self, checkpointer):
        await checkpointer.save(make_checkpoint("s1", "c1", 1, run_id="run-a"))

        assert await checkpointer.list_by_run("s1", "run-does-not-exist") == []

    async def test_scoped_to_session_even_if_run_id_reused(self, checkpointer):
        # run_id is a fresh uuid4 per call in practice, but the index is
        # keyed under session_id too — a caller-supplied run_id from a
        # different session must not leak checkpoints across sessions.
        await checkpointer.save(make_checkpoint("s1", "c1", 1, run_id="run-a"))
        await checkpointer.save(make_checkpoint("s2", "c2", 1, run_id="run-a"))

        results = await checkpointer.list_by_run("s1", "run-a")
        assert [c.metadata.checkpoint_id for c in results] == ["c1"]


class TestDelete:
    async def test_delete_specific_checkpoint(self, checkpointer):
        await checkpointer.save(make_checkpoint("s1", "c1", 1))
        await checkpointer.save(make_checkpoint("s1", "c2", 2))

        await checkpointer.delete("s1", checkpoint_id="c1")

        assert await checkpointer.load("s1", checkpoint_id="c1") is None
        assert await checkpointer.load("s1", checkpoint_id="c2") is not None

    async def test_delete_specific_checkpoint_removes_it_from_run_index(
        self, checkpointer, fake_client
    ):
        await checkpointer.save(make_checkpoint("s1", "c1", 1, run_id="run-a"))
        await checkpointer.save(make_checkpoint("s1", "c2", 2, run_id="run-a"))

        await checkpointer.delete("s1", checkpoint_id="c1")

        # Assert the raw sorted-set membership, not just `list_by_run`'s
        # output — it already tolerates a dangling id (filters out a
        # checkpoint that fails to load), so it can't tell "removed from
        # the index" apart from "index entry left dangling but unloadable".
        run_a_key = checkpointer._run_key("s1", "run-a")
        assert set(fake_client.zsets.get(run_a_key, {})) == {"c2"}

    async def test_delete_all_for_session(self, checkpointer):
        await checkpointer.save(make_checkpoint("s1", "c1", 1))
        await checkpointer.save(make_checkpoint("s1", "c2", 2))

        await checkpointer.delete("s1")

        assert await checkpointer.list("s1") == []

    async def test_delete_all_for_session_removes_run_indexes(
        self, checkpointer, fake_client
    ):
        await checkpointer.save(make_checkpoint("s1", "c1", 1, run_id="run-a"))
        await checkpointer.save(make_checkpoint("s1", "c2", 2, run_id="run-b"))
        await checkpointer.save(make_checkpoint("s2", "c3", 1, run_id="run-c"))

        await checkpointer.delete("s1")

        assert await checkpointer.list_by_run("s1", "run-a") == []
        assert await checkpointer.list_by_run("s1", "run-b") == []
        # A different session's run index is untouched.
        assert len(await checkpointer.list_by_run("s2", "run-c")) == 1
        assert not any(
            k.startswith(f"{checkpointer.prefix}session:s1:run:")
            for k in fake_client.zsets
        )


class TestClearAll:
    async def test_clear_all_removes_matching_prefix_keys(
        self, checkpointer, fake_client
    ):
        await checkpointer.save(make_checkpoint("s1", "c1", 1))

        await checkpointer.clear_all()

        assert fake_client.store == {}
        assert fake_client.zsets == {}


class TestClose:
    async def test_close_closes_underlying_client(self, checkpointer, fake_client):
        await checkpointer.load("s1")  # forces client creation
        await checkpointer.close()
        assert fake_client.closed is True


class TestCodec:
    async def test_compress_disabled_still_round_trips(self, monkeypatch, fake_client):
        cp = make_checkpointer(monkeypatch, fake_client, compress=False)
        await cp.save(make_checkpoint("s1", "c1", 1))

        loaded = await cp.load("s1", checkpoint_id="c1")
        assert loaded.state == {"step": 1}  # type: ignore

    async def test_no_compress_no_encryption_stores_state_as_plain_json(
        self, monkeypatch, fake_client
    ):
        # compress=False and no encryption_key means StateCodec.is_binary is
        # False — `state` must be embedded as a native JSON value, not
        # base64-wrapped bytes, so it stays human-readable in Redis.
        cp = make_checkpointer(monkeypatch, fake_client, compress=False)
        await cp.save(make_checkpoint("s1", "c1", 1))

        raw = json.loads(fake_client.store[cp._checkpoint_key("c1")])
        assert raw["state"] == {"step": 1}

    async def test_compress_enabled_stores_state_as_base64(self, monkeypatch, fake_client):
        cp = make_checkpointer(monkeypatch, fake_client, compress=True)
        await cp.save(make_checkpoint("s1", "c1", 1))

        raw = json.loads(fake_client.store[cp._checkpoint_key("c1")])
        assert isinstance(raw["state"], str)
        base64.b64decode(raw["state"])  # does not raise: valid base64

        loaded = await cp.load("s1", checkpoint_id="c1")
        assert loaded.state == {"step": 1}  # type: ignore

    async def test_encryption_round_trips_and_hides_plaintext(
        self, monkeypatch, fake_client
    ):
        key = Fernet.generate_key()
        cp = make_checkpointer(monkeypatch, fake_client, encryption_key=key)
        await cp.save(make_checkpoint("s1", "c1", 1))

        loaded = await cp.load("s1", checkpoint_id="c1")
        assert loaded.state == {"step": 1}  # type: ignore

        # The `state` field specifically (not the whole payload, which also
        # carries a plaintext metadata `step` counter) must be opaque
        # ciphertext once base64-decoded — not readable JSON.
        raw = json.loads(fake_client.store[cp._checkpoint_key("c1")])
        state_bytes = base64.b64decode(raw["state"])
        assert b'"step"' not in state_bytes

    async def test_oversized_state_is_rejected_even_without_compression(
        self, monkeypatch, fake_client
    ):
        # The size guard runs inside encode() unconditionally — it must not
        # be skippable by leaving compress/encryption_key off.
        cp = make_checkpointer(monkeypatch, fake_client, compress=False, max_state_bytes=10)
        with pytest.raises(CheckpointPayloadTooLargeException):
            await cp.save(make_checkpoint("s1", "c1", 1))
        assert await cp.load("s1") is None

    async def test_wrong_encryption_key_fails_to_decode(self, monkeypatch, fake_client):
        cp_write = make_checkpointer(
            monkeypatch, fake_client, encryption_key=Fernet.generate_key()
        )
        await cp_write.save(make_checkpoint("s1", "c1", 1))

        cp_read = make_checkpointer(
            monkeypatch, fake_client, encryption_key=Fernet.generate_key()
        )
        with pytest.raises(InvalidToken):
            await cp_read.load("s1", checkpoint_id="c1")


class TestRetention:
    async def test_default_is_unbounded(self, checkpointer):
        for i in range(5):
            await checkpointer.save(make_checkpoint("s1", f"c{i}", i))

        assert len(await checkpointer.list("s1", limit=100)) == 5

    async def test_max_checkpoints_per_session_drops_oldest(
        self, monkeypatch, fake_client
    ):
        cp = make_checkpointer(monkeypatch, fake_client, max_checkpoints_per_session=2)
        for i in range(5):
            await cp.save(make_checkpoint("s1", f"c{i}", i))

        results = await cp.list("s1", limit=100)
        assert {c.metadata.checkpoint_id for c in results} == {"c3", "c4"}

    async def test_dropped_checkpoints_are_unreachable_by_id(
        self, monkeypatch, fake_client
    ):
        cp = make_checkpointer(monkeypatch, fake_client, max_checkpoints_per_session=1)
        await cp.save(make_checkpoint("s1", "c0", 0))
        await cp.save(make_checkpoint("s1", "c1", 1))

        assert await cp.load("s1", checkpoint_id="c0") is None
        assert await cp.load("s1", checkpoint_id="c1") is not None

    async def test_dropped_checkpoints_are_unindexed_from_their_run(
        self, monkeypatch, fake_client
    ):
        cp = make_checkpointer(monkeypatch, fake_client, max_checkpoints_per_session=1)
        await cp.save(make_checkpoint("s1", "c0", 0, run_id="run-a"))
        await cp.save(make_checkpoint("s1", "c1", 1, run_id="run-b"))

        # "c0" was evicted by the cap — assert directly against the raw
        # sorted-set membership, not just `list_by_run`'s output: that
        # method already tolerates a dangling id (filters out a checkpoint
        # that fails to load), so it would report "run-a" empty either way
        # and wouldn't actually prove the index entry was removed.
        run_a_key = cp._run_key("s1", "run-a")
        run_b_key = cp._run_key("s1", "run-b")
        assert "c0" not in fake_client.zsets.get(run_a_key, {})
        assert "c1" in fake_client.zsets.get(run_b_key, {})
