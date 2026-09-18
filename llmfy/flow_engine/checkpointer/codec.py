"""Wire-format layer for checkpoint storage: dict <-> bytes.

This is deliberately separate from `serde.py`, which owns a different job
(the type-registry allow-list turning custom objects into/out of a
JSON-safe dict). `StateCodec` only ever sees an already JSON-safe dict — it
owns compressing and optionally encrypting the bytes that actually get
written to a SQL/Redis backend, nothing about object reconstruction.
"""

from __future__ import annotations

import json
import zlib
from typing import Any

from llmfy.exception.llmfy_exception import (
    CheckpointPayloadTooLargeException,
    LLMfyException,
)

DEFAULT_MAX_STATE_BYTES = 25 * 1024 * 1024  # 25 MiB


class StateCodec:
    """Encodes a JSON-safe state dict to bytes for storage, and back.

    Order is compress-then-encrypt on encode (reversed on decode): ciphertext
    doesn't compress, so compression only helps if it runs first.
    """

    def __init__(
        self,
        *,
        compress: bool = True,
        encryption_key: bytes | str | None = None,
        max_state_bytes: int | None = DEFAULT_MAX_STATE_BYTES,
    ):
        self._compress = compress
        self._max_state_bytes = max_state_bytes
        self._fernet = self._build_fernet(encryption_key) if encryption_key else None

    @property
    def is_binary(self) -> bool:
        """True when `encode()`'s output is compressed and/or encrypted
        bytes that need a binary-safe transport (e.g. base64 for a
        text-mode Redis payload) — False when it's exactly
        `json.dumps(state).encode("utf-8")`, i.e. plain UTF-8 JSON that a
        caller can embed directly instead of wrapping it further."""
        return self._compress or self._fernet is not None

    @staticmethod
    def _build_fernet(key: bytes | str):
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:
            raise LLMfyException(
                "The 'cryptography' package is required for checkpoint "
                "encryption.\nInstall with: pip install \"llmfy[crypto]\""
            ) from exc
        return Fernet(key)

    def encode(self, state: dict[str, Any], *, session_id: str = "") -> bytes:
        raw = json.dumps(state).encode("utf-8")
        if self._max_state_bytes is not None and len(raw) > self._max_state_bytes:
            raise CheckpointPayloadTooLargeException(
                f"Checkpoint state for session '{session_id}' is {len(raw)} "
                f"bytes, exceeding the configured limit of "
                f"{self._max_state_bytes} bytes.",
                session_id=session_id,
                size_bytes=len(raw),
                max_bytes=self._max_state_bytes,
            )
        if self._compress:
            raw = zlib.compress(raw)
        if self._fernet is not None:
            raw = self._fernet.encrypt(raw)
        return raw

    def decode(self, data: bytes) -> dict[str, Any]:
        raw = data
        if self._fernet is not None:
            raw = self._fernet.decrypt(raw)
        if self._compress:
            raw = zlib.decompress(raw)
        return json.loads(raw)
