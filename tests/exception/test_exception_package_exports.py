"""Verifies llmfy/exception/__init__.py's public re-export surface."""

import llmfy.exception as exception_pkg
from llmfy.exception.llmfy_exception import LLMfyException


def test_all_matches_actual_importable_names():
    for name in exception_pkg.__all__:
        assert hasattr(exception_pkg, name), f"{name} listed in __all__ but not importable"


def test_all_exported_names_are_llmfy_exception_subclasses():
    for name in exception_pkg.__all__:
        obj = getattr(exception_pkg, name)
        assert issubclass(obj, LLMfyException)


def test_timeout_type_is_not_exported_at_package_level():
    # TimeoutType lives in llmfy_exception.py but is intentionally not
    # re-exported from the `llmfy.exception` package level.
    assert "TimeoutType" not in exception_pkg.__all__
    assert not hasattr(exception_pkg, "TimeoutType")


def test_no_stray_extra_exports():
    # Guards against silently adding a name to the module without updating
    # __all__ (or vice versa) going unnoticed.
    assert set(exception_pkg.__all__) == {
        "LLMfyException",
        "AuthenticationException",
        "ContentFilterException",
        "InvalidRequestException",
        "ModelErrorException",
        "ModelNotFoundException",
        "PermissionDeniedException",
        "QuotaExceededException",
        "RateLimitException",
        "ServiceUnavailableException",
        "TimeoutException",
        "GraphValidationException",
        "NodeExecutionException",
        "NodeTimeoutException",
        "StepLimitExceededException",
        "CheckpointDeserializationException",
        "CheckpointPayloadTooLargeException",
        "InvalidSessionIdException",
    }
