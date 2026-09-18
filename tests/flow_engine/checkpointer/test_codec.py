"""Unit tests for llmfy/flow_engine/checkpointer/codec.py.

`StateCodec` is the wire-format boundary between an already JSON-safe state
dict (produced by `serde.py`) and the bytes a SQL/Redis checkpointer backend
actually writes. Tests cover every compress/encrypt combination round-trips,
the size guard fails closed, and a missing `cryptography` install surfaces a
clear, actionable error rather than an opaque ImportError.
"""

import sys

import pytest
from cryptography.fernet import Fernet

from llmfy.exception.llmfy_exception import (
    CheckpointPayloadTooLargeException,
    LLMfyException,
)
from llmfy.flow_engine.checkpointer.codec import StateCodec

STATE = {"messages": ["hello", "world"], "count": 2, "nested": {"a": [1, 2, 3]}}


class TestRoundTrip:
    def test_plain(self):
        codec = StateCodec(compress=False, encryption_key=None)
        assert codec.decode(codec.encode(STATE)) == STATE

    def test_compress_only(self):
        codec = StateCodec(compress=True, encryption_key=None)
        assert codec.decode(codec.encode(STATE)) == STATE

    def test_encrypt_only(self):
        key = Fernet.generate_key()
        codec = StateCodec(compress=False, encryption_key=key)
        assert codec.decode(codec.encode(STATE)) == STATE

    def test_compress_and_encrypt(self):
        key = Fernet.generate_key()
        codec = StateCodec(compress=True, encryption_key=key)
        assert codec.decode(codec.encode(STATE)) == STATE

    def test_encrypted_output_is_not_readable_plaintext(self):
        key = Fernet.generate_key()
        codec = StateCodec(compress=False, encryption_key=key)
        encoded = codec.encode(STATE)
        assert b"hello" not in encoded

    def test_accepts_str_key_like_generate_key_output_decoded(self):
        key = Fernet.generate_key().decode("ascii")
        codec = StateCodec(compress=False, encryption_key=key)
        assert codec.decode(codec.encode(STATE)) == STATE


class TestSizeGuard:
    def test_raises_when_over_limit(self):
        codec = StateCodec(compress=False, max_state_bytes=10)
        with pytest.raises(CheckpointPayloadTooLargeException) as exc_info:
            codec.encode(STATE, session_id="s1")
        assert exc_info.value.session_id == "s1"
        assert exc_info.value.max_bytes == 10
        assert exc_info.value.size_bytes > 10

    def test_none_disables_the_check(self):
        codec = StateCodec(compress=False, max_state_bytes=None)
        codec.encode(STATE)  # does not raise

    def test_default_limit_allows_small_state(self):
        codec = StateCodec(compress=False)
        codec.encode(STATE)  # does not raise


class TestMissingCryptographyDependency:
    def test_encryption_key_without_cryptography_raises_clear_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cryptography.fernet", None)
        with pytest.raises(LLMfyException, match="llmfy\\[crypto\\]"):
            StateCodec(encryption_key=b"not-a-real-key")

    def test_no_encryption_key_never_imports_cryptography(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cryptography.fernet", None)
        StateCodec(encryption_key=None)  # does not raise
