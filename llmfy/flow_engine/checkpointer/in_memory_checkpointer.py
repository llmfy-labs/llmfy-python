from collections import defaultdict
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any

from llmfy.flow_engine.checkpointer.base_checkpointer import (
    BaseCheckpointer,
    Checkpoint,
)


class InMemoryCheckpointer(BaseCheckpointer):
    """In-memory checkpoint storage backend.

    Keeps live Python objects (via `deepcopy`) rather than crossing a
    serialization boundary, so `state` never needs to go through
    `checkpointer/serde.py`'s type registry — arbitrary objects in state
    work here without being registered. For the same reason, it has no
    `compress`/`encryption_key` options like `SQLCheckpointer`/
    `RedisCheckpointer`: state here never leaves process memory as bytes,
    so there is nothing to compress or encrypt.
    """

    requires_serialization = False

    def __init__(
        self,
        max_checkpoints_per_session: int | None = None,
        ttl_seconds: int | None = None,
    ):
        """Initialize the memory checkpointer.

        Args:
            max_checkpoints_per_session: If set, only the newest N
                checkpoints are retained per session — older ones are
                dropped on save. `None` (default) keeps every checkpoint
                ever saved, matching the pre-existing unbounded behavior.
            ttl_seconds: If set, checkpoints older than this age are dropped
                on save (checked against wall-clock time, since this backend
                has no native expiry). Independent of, and combinable with,
                `max_checkpoints_per_session`.
        """
        # Storage: session_id -> list of checkpoints
        self._storage: dict[str, list[Checkpoint]] = defaultdict(list)
        # Index: checkpoint_id -> (session_id, checkpoint)
        self._index: dict[str, tuple[str, Checkpoint]] = {}
        self.max_checkpoints_per_session = max_checkpoints_per_session
        self.ttl_seconds = ttl_seconds

    async def save(self, checkpoint: Checkpoint) -> None:
        """
        Save a checkpoint to memory.

        Args:
            checkpoint: The checkpoint to save
        """
        # Deep copy to prevent external modifications
        checkpoint_copy = deepcopy(checkpoint)

        session_id = checkpoint.metadata.session_id
        checkpoint_id = checkpoint.metadata.checkpoint_id

        # Add to storage
        self._storage[session_id].append(checkpoint_copy)

        # Sort by created_at (newest first)
        self._storage[session_id].sort(
            key=lambda c: c.metadata.created_at,
            reverse=True
        )

        # Add to index
        self._index[checkpoint_id] = (session_id, checkpoint_copy)

        # Retention: drop checkpoints older than the configured TTL
        if self.ttl_seconds is not None:
            cutoff = datetime.now(UTC) - timedelta(seconds=self.ttl_seconds)
            kept, expired = [], []
            for c in self._storage[session_id]:
                (expired if c.metadata.created_at < cutoff else kept).append(c)
            if expired:
                self._storage[session_id] = kept
                for dropped in expired:
                    self._index.pop(dropped.metadata.checkpoint_id, None)

        # Retention: drop the oldest checkpoints beyond the configured cap
        if self.max_checkpoints_per_session is not None:
            overflow = self._storage[session_id][self.max_checkpoints_per_session:]
            if overflow:
                self._storage[session_id] = self._storage[session_id][
                    : self.max_checkpoints_per_session
                ]
                for dropped in overflow:
                    self._index.pop(dropped.metadata.checkpoint_id, None)
    
    async def load(self, session_id: str, checkpoint_id: str | None = None) -> Checkpoint | None:
        """
        Load a checkpoint from memory.
        
        Args:
            session_id: The session ID
            checkpoint_id: Specific checkpoint ID, or None for latest
            
        Returns:
            The checkpoint if found, None otherwise
        """
        if checkpoint_id:
            # Load specific checkpoint
            if checkpoint_id in self._index:
                stored_session_id, checkpoint = self._index[checkpoint_id]
                if stored_session_id == session_id:
                    return deepcopy(checkpoint)
            return None
        else:
            # Load latest checkpoint for thread
            if session_id in self._storage and self._storage[session_id]:
                return deepcopy(self._storage[session_id][0])
            return None
    
    async def list(self, session_id: str, limit: int = 10) -> list[Checkpoint]:
        """
        List checkpoints for a thread.
        
        Args:
            session_id: The session ID
            limit: Maximum number of checkpoints to return
            
        Returns:
            List of checkpoints, newest first
        """
        if session_id not in self._storage:
            return []
        
        checkpoints = self._storage[session_id][:limit]
        return [deepcopy(c) for c in checkpoints]
    
    async def delete(self, session_id: str, checkpoint_id: str | None = None) -> None:
        """
        Delete checkpoint(s) from memory.
        
        Args:
            session_id: The session ID
            checkpoint_id: Specific checkpoint ID, or None to delete all for session
        """
        if checkpoint_id:
            # Delete specific checkpoint
            if checkpoint_id in self._index:
                stored_session_id, checkpoint = self._index[checkpoint_id]
                if stored_session_id == session_id:
                    # Remove from storage
                    self._storage[session_id] = [
                        c for c in self._storage[session_id]
                        if c.metadata.checkpoint_id != checkpoint_id
                    ]
                    # Remove from index
                    del self._index[checkpoint_id]
                    
                    # Clean up empty thread storage
                    if not self._storage[session_id]:
                        del self._storage[session_id]
        else:
            # Delete all checkpoints for thread
            if session_id in self._storage:
                # Remove from index
                for checkpoint in self._storage[session_id]:
                    checkpoint_id = checkpoint.metadata.checkpoint_id
                    if checkpoint_id in self._index:
                        del self._index[checkpoint_id]
                
                # Remove from storage
                del self._storage[session_id]
    
    async def clear_all(self) -> None:
        """Clear all checkpoints from memory."""
        self._storage.clear()
        self._index.clear()
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_sessions": len(self._storage),
            "total_checkpoints": len(self._index),
            "checkpoints_per_session": {
                session_id: len(checkpoints)
                for session_id, checkpoints in self._storage.items()
            }
        }