from __future__ import annotations

import base64
import json

from llmfy.exception.llmfy_exception import LLMfyException
from llmfy.flow_engine.checkpointer.base_checkpointer import (
    BaseCheckpointer,
    Checkpoint,
)
from llmfy.flow_engine.checkpointer.codec import DEFAULT_MAX_STATE_BYTES, StateCodec

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisCheckpointer(BaseCheckpointer):
    """Redis checkpoint storage backend."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        prefix: str = "llmfy_checkpoint:",
        ttl: int | None = None,
        compress: bool = False,
        encryption_key: bytes | str | None = None,
        max_state_bytes: int | None = DEFAULT_MAX_STATE_BYTES,
        max_checkpoints_per_session: int | None = None,
    ):
        """
        Initialize the Redis checkpointer.

        Args:
            redis_url: Redis connection URL
            prefix: Key prefix for checkpoints
            ttl: Sliding, whole-session expiration in seconds (None = no
                expiration). Refreshed via `EXPIRE` on every `save()`, so
                it bounds how long an *abandoned* session's checkpoints
                survive — it does NOT prune individual old checkpoints
                out of an active session (use `max_checkpoints_per_session`
                for that).
            compress: zlib-compress the serialized state before writing.
                Off by default. When off (and `encryption_key` is unset),
                `state` is stored as plain, human-readable JSON rather than
                base64-wrapped bytes.
            encryption_key: optional Fernet key (see
                `cryptography.fernet.Fernet.generate_key()`) to encrypt state
                at rest. Requires `pip install "llmfy[crypto]"`. `None`
                (default) stores state unencrypted.
            max_state_bytes: reject a checkpoint whose serialized state
                exceeds this many bytes, raising
                `CheckpointPayloadTooLargeException`, instead of writing an
                unbounded payload. `None` disables the check.
            max_checkpoints_per_session: if set, only the newest N
                checkpoints are retained per session — older ones (and their
                sorted-set entries) are deleted right after each save.
        """
        if not REDIS_AVAILABLE:
            raise LLMfyException(
                "redis package is not installed. redis package is required for RedisCheckpointer. "
                'Install it using `pip install "llmfy[redis]"`'
            )

        self.redis_url = redis_url
        self.prefix = prefix
        self.ttl = ttl
        self._codec = StateCodec(
            compress=compress,
            encryption_key=encryption_key,
            max_state_bytes=max_state_bytes,
        )
        self.max_checkpoints_per_session = max_checkpoints_per_session
        self._client: redis.Redis | None = None

    async def _get_client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            self._client = await redis.from_url(
                self.redis_url, encoding="utf-8", decode_responses=True
            )
        return self._client

    def _session_key(self, session_id: str) -> str:
        """Get Redis key for session's checkpoint list."""
        return f"{self.prefix}session:{session_id}"

    def _checkpoint_key(self, checkpoint_id: str) -> str:
        """Get Redis key for specific checkpoint."""
        return f"{self.prefix}checkpoint:{checkpoint_id}"

    def _run_key(self, session_id: str, run_id: str) -> str:
        """Get Redis key for one run's (one `invoke()`/`stream()` call's)
        checkpoint list. Nested under the session's own key namespace
        (`session:{session_id}:run:{run_id}`, not a separate `run:`
        top-level namespace) so `SCAN MATCH {prefix}session:{session_id}*`
        finds every key belonging to a session in one pattern — handy when
        browsing Redis directly — and so `delete(session_id)` can clean
        every run index for that session via a single `SCAN` (see
        `_run_key_pattern`)."""
        return f"{self.prefix}session:{session_id}:run:{run_id}"

    def _run_key_pattern(self, session_id: str) -> str:
        """`SCAN`-compatible pattern matching every run key under a session."""
        return f"{self.prefix}session:{session_id}:run:*"

    async def save(self, checkpoint: Checkpoint) -> None:
        """
        Save a checkpoint to Redis.

        Args:
            checkpoint: The checkpoint to save
        """
        client = await self._get_client()

        session_id = checkpoint.metadata.session_id
        checkpoint_id = checkpoint.metadata.checkpoint_id
        timestamp = checkpoint.metadata.created_at.timestamp()

        # Save checkpoint data. Metadata stays plain JSON (small, not
        # sensitive, and needed as-is for the sorted-set/listing paths).
        # `state` always runs through the codec (so the size guard always
        # applies), but only gets base64-wrapped when the codec actually
        # produced binary output (compressed and/or encrypted) — with
        # compress=False and no encryption_key, encode() is just UTF-8 JSON,
        # so it's embedded directly as a plain, human-readable JSON value
        # instead of being needlessly obscured behind base64.
        checkpoint_key = self._checkpoint_key(checkpoint_id)
        payload = checkpoint.to_dict()
        encoded_state = self._codec.encode(checkpoint.state, session_id=session_id)
        if self._codec.is_binary:
            payload["state"] = base64.b64encode(encoded_state).decode("ascii")
        else:
            payload["state"] = json.loads(encoded_state)
        checkpoint_data = json.dumps(payload)

        await client.set(checkpoint_key, checkpoint_data)

        if self.ttl:
            await client.expire(checkpoint_key, self.ttl)

        # Add to thread's sorted set (sorted by timestamp)
        session_key = self._session_key(session_id)
        await client.zadd(session_key, {checkpoint_id: timestamp})

        if self.ttl:
            await client.expire(session_key, self.ttl)

        # Add to this run's sorted set too, so `list_by_run` can find every
        # checkpoint saved during one `invoke()`/`stream()` call without
        # loading the whole session's history.
        run_key = self._run_key(session_id, checkpoint.metadata.run_id)
        await client.zadd(run_key, {checkpoint_id: timestamp})

        if self.ttl:
            await client.expire(run_key, self.ttl)

        if self.max_checkpoints_per_session is not None:
            await self._enforce_count_cap(client, session_key, session_id)

    async def _enforce_count_cap(
        self, client: redis.Redis, session_key: str, session_id: str
    ) -> None:
        """Trim a session's sorted set down to the newest
        `max_checkpoints_per_session` entries, deleting the dropped
        checkpoints' data keys along with their sorted-set membership (in
        both the session index and whichever run index each one belongs
        to)."""
        n = self.max_checkpoints_per_session
        assert n is not None
        overflow_ids = await client.zrevrange(session_key, n, -1)
        if not overflow_ids:
            return
        overflow_keys = [self._checkpoint_key(cid) for cid in overflow_ids]
        for checkpoint_id, checkpoint_key in zip(overflow_ids, overflow_keys, strict=True):
            await self._unindex_run(client, session_id, checkpoint_key, checkpoint_id)
        await client.delete(*overflow_keys)
        await client.zrem(session_key, *overflow_ids)

    async def _unindex_run(
        self,
        client: redis.Redis,
        session_id: str,
        checkpoint_key: str,
        checkpoint_id: str,
    ) -> None:
        """Remove one checkpoint's entry from its run index, read from the
        checkpoint's own stored data. A no-op if the checkpoint is already
        gone (nothing to read `run_id` from)."""
        raw = await client.get(checkpoint_key)
        if raw is None:
            return
        run_id = json.loads(raw).get("run_id")
        if run_id:
            await client.zrem(self._run_key(session_id, run_id), checkpoint_id)

    async def load(
        self,
        session_id: str,
        checkpoint_id: str | None = None,
    ) -> Checkpoint | None:
        """
        Load a checkpoint from Redis.

        Args:
            session_id: The session ID
            checkpoint_id: Specific checkpoint ID, or None for latest

        Returns:
            The checkpoint if found, None otherwise
        """
        client = await self._get_client()

        if checkpoint_id is None:
            # Get latest checkpoint ID from sorted set
            session_key = self._session_key(session_id)
            results = await client.zrevrange(session_key, 0, 0)

            if not results:
                return None

            checkpoint_id = results[0]

        # Load checkpoint data
        checkpoint_key = self._checkpoint_key(checkpoint_id)  # type: ignore
        data = await client.get(checkpoint_key)

        if data is None:
            return None

        checkpoint_dict = json.loads(data)
        if self._codec.is_binary:
            checkpoint_dict["state"] = self._codec.decode(
                base64.b64decode(checkpoint_dict["state"])
            )
        # else: already the plain state dict, embedded as-is by save().
        return Checkpoint.from_dict(checkpoint_dict)

    async def list(self, session_id: str, limit: int = 10) -> list[Checkpoint]:
        """
        List checkpoints for a thread.

        Args:
            session_id: The session ID
            limit: Maximum number of checkpoints to return

        Returns:
            List of checkpoints, newest first
        """
        client = await self._get_client()

        # Get checkpoint IDs from sorted set (newest first)
        thread_key = self._session_key(session_id)
        checkpoint_ids = await client.zrevrange(thread_key, 0, limit - 1)

        if not checkpoint_ids:
            return []

        # Load all checkpoints
        checkpoints = []
        for checkpoint_id in checkpoint_ids:
            checkpoint = await self.load(session_id, checkpoint_id)
            if checkpoint:
                checkpoints.append(checkpoint)

        return checkpoints

    async def list_by_run(
        self, session_id: str, run_id: str, limit: int = 100
    ) -> list[Checkpoint]:
        """
        List checkpoints saved during one run — one `invoke()`/`stream()`
        call, fresh or resumed (see `CheckpointMetadata.run_id`) — newest
        first. Unlike `list()`, which walks a session's whole checkpoint
        history, this reads only that run's own index, so it stays cheap
        even for a long-lived session with many past runs.

        Args:
            session_id: The session ID
            run_id: The run ID to filter by
            limit: Maximum number of checkpoints to return

        Returns:
            List of checkpoints from that run, newest first
        """
        client = await self._get_client()

        run_key = self._run_key(session_id, run_id)
        checkpoint_ids = await client.zrevrange(run_key, 0, limit - 1)

        if not checkpoint_ids:
            return []

        checkpoints = []
        for checkpoint_id in checkpoint_ids:
            checkpoint = await self.load(session_id, checkpoint_id)
            if checkpoint:
                checkpoints.append(checkpoint)

        return checkpoints

    async def delete(self, session_id: str, checkpoint_id: str | None = None) -> None:
        """
        Delete checkpoint(s) from Redis.

        Args:
            session_id: The session ID
            checkpoint_id: Specific checkpoint ID, or None to delete all for thread
        """
        client = await self._get_client()
        session_key = self._session_key(session_id)

        if checkpoint_id:
            # Delete specific checkpoint
            checkpoint_key = self._checkpoint_key(checkpoint_id)
            await self._unindex_run(client, session_id, checkpoint_key, checkpoint_id)
            await client.delete(checkpoint_key)
            await client.zrem(session_key, checkpoint_id)
        else:
            # Delete all checkpoints for thread
            checkpoint_ids = await client.zrange(session_key, 0, -1)

            if checkpoint_ids:
                # Delete all checkpoint data
                checkpoint_keys = [self._checkpoint_key(cid) for cid in checkpoint_ids]
                await client.delete(*checkpoint_keys)

            # Delete session sorted set
            await client.delete(session_key)

            # Delete every run index under this session — cheaper than
            # tracking each checkpoint's run_id individually since we're
            # already wiping the whole session.
            await self._delete_run_indexes(client, session_id)

    async def _delete_run_indexes(self, client: redis.Redis, session_id: str) -> None:
        """Delete every run index key under one session, via `SCAN` — the
        same technique `clear_all()` uses, scoped to this session's run
        keys instead of every key under `prefix`."""
        pattern = self._run_key_pattern(session_id)
        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor, match=pattern, count=100)
            if keys:
                await client.delete(*keys)
            if cursor == 0:
                break

    async def clear_all(self) -> None:
        """Clear all checkpoints from Redis."""
        client = await self._get_client()

        # Find all keys with our prefix
        pattern = f"{self.prefix}*"
        cursor = 0

        while True:
            cursor, keys = await client.scan(cursor, match=pattern, count=100)

            if keys:
                await client.delete(*keys)

            if cursor == 0:
                break

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
