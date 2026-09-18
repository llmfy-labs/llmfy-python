from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint.

    `run_id` identifies the single `invoke()`/`stream()` call that produced
    this checkpoint — every checkpoint saved during that one call shares it,
    so checkpoints can be grouped by run to see which nodes executed
    together, distinct from `session_id` which spans every call (fresh and
    resumed) across a session's lifetime. `step` is this checkpoint's
    position within that run (always starting at 0 for a new run, even a
    resumed one), not a cumulative count across the session.

    `node` is the node whose completion produced this checkpoint. `prev_node`
    is whichever node's completion led to `node` running — `START`
    (`"__start__"`) if `node` is the first node executed in this run (fresh
    or resumed: a resumed call gets its own fresh `run_id` and so, from that
    run's own perspective, "starts" at its resume point). A node reached via
    a dynamic (`Send`) fan-out has the dispatching node as `prev_node`, same
    as any other successor.

    `updated_fields` are the state keys this checkpoint's update touched
    (the node's own returned keys; for a dynamic fan-out's single
    aggregate checkpoint, the union of every branch's keys) — a cheap
    "what changed here" without diffing full state snapshots.

    `attempt` is the 1-indexed try that succeeded (1 = no retry needed).
    `None` for a dynamic fan-out's aggregate checkpoint, which doesn't
    represent one node's one execution — several branches, each with its
    own possibly-different attempt count, are committed together.

    `dispatch_id` identifies a dynamic (`Send`) fan-out dispatch — set only
    on the single checkpoint saved for that dispatch (correlates with the
    same `dispatch_id` on that dispatch's `FlowEngineStreamResponse`
    events); `None` for every other checkpoint.
    """
    checkpoint_id: str
    session_id: str
    run_id: str
    created_at: datetime
    node: str
    prev_node: str
    step: int
    updated_fields: list[str]
    attempt: int | None
    dispatch_id: str | None


@dataclass
class Checkpoint:
    """A saved workflow state checkpoint.

    `state` is always an already-JSON-safe dict by the time it reaches a
    checkpointer backend. Turning custom objects into/out of that dict is
    owned exclusively by `llmfy.flow_engine.checkpointer.serde`, driven by
    the `FlowEngine`'s type registry — `Checkpoint` itself does no
    reflection and reconstructs nothing beyond its own metadata fields.
    """
    metadata: CheckpointMetadata
    state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert checkpoint to a plain dict for storage (e.g. json.dumps)."""
        return {
            "checkpoint_id": self.metadata.checkpoint_id,
            "session_id": self.metadata.session_id,
            "run_id": self.metadata.run_id,
            "created_at": self.metadata.created_at.isoformat(),
            "node": self.metadata.node,
            "prev_node": self.metadata.prev_node,
            "step": self.metadata.step,
            "updated_fields": self.metadata.updated_fields,
            "attempt": self.metadata.attempt,
            "dispatch_id": self.metadata.dispatch_id,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        """Create a checkpoint from a dict produced by `to_dict`."""
        metadata = CheckpointMetadata(
            checkpoint_id=data["checkpoint_id"],
            session_id=data["session_id"],
            run_id=data["run_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            node=data["node"],
            prev_node=data["prev_node"],
            step=data["step"],
            updated_fields=data["updated_fields"],
            attempt=data["attempt"],
            dispatch_id=data["dispatch_id"],
        )
        return cls(metadata=metadata, state=data["state"])


class BaseCheckpointer(ABC):
    """Base class for checkpoint storage backends.

    `requires_serialization` tells `FlowEngine` whether crossing this
    backend's save/load boundary needs `checkpointer/serde.py`'s
    allow-listed (de)serialization. `True` (the default) is the safe
    choice for any backend that might persist outside the current
    process — subclasses that keep live Python objects in-process (like
    `InMemoryCheckpointer`) can opt out by overriding it to `False`.
    """

    requires_serialization: bool = True

    @abstractmethod
    async def save(self, checkpoint: Checkpoint) -> None:
        """
        Save a checkpoint.

        Args:
            checkpoint: The checkpoint to save
        """
        pass

    @abstractmethod
    async def load(self, session_id: str, checkpoint_id: str | None = None) -> Checkpoint | None:
        """
        Load a checkpoint.

        Args:
            session_id: The session ID
            checkpoint_id: Specific checkpoint ID, or None for latest

        Returns:
            The checkpoint if found, None otherwise
        """
        pass

    @abstractmethod
    async def list(self, session_id: str, limit: int = 10) -> list[Checkpoint]:
        """
        List checkpoints for a thread.

        Args:
            session_id: The session ID
            limit: Maximum number of checkpoints to return

        Returns:
            List of checkpoints, newest first
        """
        pass

    @abstractmethod
    async def delete(self, session_id: str, checkpoint_id: str | None = None) -> None:
        """
        Delete checkpoint(s).

        Args:
            session_id: The session ID
            checkpoint_id: Specific checkpoint ID, or None to delete all for thread
        """
        pass

    @abstractmethod
    async def clear_all(self) -> None:
        """Clear all checkpoints from storage."""
        pass
