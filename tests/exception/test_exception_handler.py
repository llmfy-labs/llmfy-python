"""Unit tests for llmfy/exception/exception_handler.py.

`handle_openai_error`/`handle_anthropic_error` dispatch purely on
`type(e).__name__` plus `hasattr` checks — never `isinstance` against the real
SDK classes — so those two are tested with small locally-defined fake
exception classes carrying the real SDK's class names. This tests OUR
dispatch logic directly rather than the SDK's own exception construction
(out of scope: that's the provider SDK's code, not ours).

`handle_bedrock_error` and `handle_google_error` DO use `isinstance` against
real SDK exception types, so those are tested against real `botocore`/`httpx2`/
`google-genai` exception instances (all installed via the `dev`/`test`
dependency groups — no network, no real credentials needed to construct them).

Timeout tests take the shared `httpx_mod` fixture (tests/conftest.py), which
runs them once per installed httpx-family package (`httpx2` and/or `httpx`): current provider SDKs raise
`httpx2` exceptions, older ones `httpx`, and the handlers must map both.
"""

from botocore.exceptions import ClientError, ConnectTimeoutError, ReadTimeoutError
from google.genai import errors as google_errors

from llmfy.exception.exception_handler import (
    handle_anthropic_error,
    handle_bedrock_error,
    handle_google_error,
    handle_openai_error,
)
from llmfy.exception.llmfy_exception import (
    AuthenticationException,
    InvalidRequestException,
    LLMfyException,
    ModelErrorException,
    ModelNotFoundException,
    RateLimitException,
    ServiceUnavailableException,
    TimeoutException,
    TimeoutType,
)
from llmfy.llmfy_core.service_provider import ServiceProvider


def _fake_exception(name: str, message: str = "boom", **attrs):
    """Build an exception instance whose class name matches a real SDK
    exception (e.g. "RateLimitError") without depending on that SDK's actual
    constructor signature — handle_openai_error/handle_anthropic_error only
    ever inspect `type(e).__name__` and `hasattr`/`getattr`.
    """
    cls = type(name, (Exception,), {})
    exc = cls(message)
    for key, value in attrs.items():
        setattr(exc, key, value)
    return exc


class FakeGoogleAPIError(google_errors.APIError):
    """Subclass of the real `google.genai.errors.APIError` that sets its
    attributes directly instead of going through the real constructor (whose
    exact signature isn't part of our contract to depend on) — `isinstance`
    checks in `handle_google_error` still pass since this is a real subclass.
    """

    def __init__(self, code: int, message: str, details=None):
        self.code = code
        self.message = message
        if details is not None:
            self.details = details


class TestHandleBedrockError:
    def test_read_timeout_maps_to_timeout_exception(self):
        e = ReadTimeoutError(
            endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"
        )
        result = handle_bedrock_error(e)
        assert isinstance(result, TimeoutException)
        assert result.timeout_type == TimeoutType.READ
        assert result.provider == ServiceProvider.BEDROCK
        assert result.raw_error is e

    def test_connect_timeout_maps_to_timeout_exception(self):
        e = ConnectTimeoutError(
            endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"
        )
        result = handle_bedrock_error(e)
        assert isinstance(result, TimeoutException)
        assert result.timeout_type == TimeoutType.CONNECT

    def test_non_client_error_becomes_plain_llmfy_exception(self):
        result = handle_bedrock_error(ValueError("something unrelated"))
        assert type(result) is LLMfyException
        assert result.provider == ServiceProvider.BEDROCK
        assert result.status_code is None

    def test_mapped_client_error_code(self):
        e = ClientError(
            {
                "Error": {
                    "Code": "ThrottlingException",
                    "Message": "Too many requests",
                },
                "ResponseMetadata": {"HTTPStatusCode": 429},
            },
            "Converse",
        )
        result = handle_bedrock_error(e)
        assert isinstance(result, RateLimitException)
        assert result.status_code == 429
        assert result.message == "Too many requests"
        assert result.provider == ServiceProvider.BEDROCK

    def test_http_status_from_response_metadata_overrides_default(self):
        # HTTPStatusCode is present and non-zero -> it wins over the map's
        # own default status (503) even though they happen to differ here.
        e = ClientError(
            {
                "Error": {"Code": "ServiceUnavailableException", "Message": "down"},
                "ResponseMetadata": {"HTTPStatusCode": 599},
            },
            "Converse",
        )
        result = handle_bedrock_error(e)
        assert result.status_code == 599

    def test_missing_http_status_falls_back_to_map_default(self):
        e = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
            "Converse",
        )
        result = handle_bedrock_error(e)
        assert result.status_code == 429  # BEDROCK_ERROR_MAP default

    def test_unknown_error_code_falls_back_to_plain_llmfy_exception(self):
        e = ClientError(
            {
                "Error": {"Code": "SomeBrandNewException", "Message": "unmapped"},
                "ResponseMetadata": {"HTTPStatusCode": 599},
            },
            "Converse",
        )
        result = handle_bedrock_error(e)
        assert type(result) is LLMfyException
        assert result.status_code == 599

    def test_missing_message_falls_back_to_str_of_exception(self):
        e = ClientError({"Error": {"Code": "ValidationException"}}, "Converse")
        result = handle_bedrock_error(e)
        assert isinstance(result, InvalidRequestException)
        assert result.message == str(e)

    def test_model_timeout_code_produces_timeout_exception_with_model_type(self):
        # Distinct from the ReadTimeoutError/ConnectTimeoutError isinstance
        # branches above: this is a *mapped ClientError code*, not a raw
        # botocore timeout exception, yet still resolves to TimeoutException.
        e = ClientError(
            {
                "Error": {"Code": "ModelTimeoutException", "Message": "model too slow"},
                "ResponseMetadata": {"HTTPStatusCode": 408},
            },
            "Converse",
        )
        result = handle_bedrock_error(e)
        assert isinstance(result, TimeoutException)
        assert result.timeout_type == TimeoutType.MODEL
        assert result.raw_error == e.response


class TestHandleOpenAIError:
    def test_mapped_error_type(self):
        e = _fake_exception("RateLimitError", "rate limited")
        result = handle_openai_error(e)
        assert isinstance(result, RateLimitException)
        assert result.status_code == 429
        assert result.provider == ServiceProvider.OPENAI
        assert result.raw_error["type"] == "RateLimitError"  # type: ignore
        assert result.raw_error["message"] == "rate limited"  # type: ignore

    def test_unmapped_error_type_falls_back_to_plain_llmfy_exception(self):
        e = _fake_exception("SomeWeirdError", "?")
        result = handle_openai_error(e)
        assert type(result) is LLMfyException

    def test_explicit_status_code_wins_over_map_default(self):
        e = _fake_exception("BadRequestError", "bad", status_code=499)
        result = handle_openai_error(e)
        assert result.status_code == 499

    def test_falsy_status_code_falls_back_to_map_default(self):
        e = _fake_exception("BadRequestError", "bad", status_code=None)
        result = handle_openai_error(e)
        assert result.status_code == 400  # OPENAI_ERROR_MAP default

    def test_response_request_id_and_body_are_collected_when_present(self):
        e = _fake_exception(
            "NotFoundError",
            "missing",
            response={"raw": "resp"},
            request_id="req_123",
            body={"detail": "x"},
        )
        result = handle_openai_error(e)
        assert result.raw_error["response"] == {"raw": "resp"}  # type: ignore
        assert result.raw_error["request_id"] == "req_123"  # type: ignore
        assert result.raw_error["body"] == {"detail": "x"}  # type: ignore

    def test_absent_optional_attrs_are_not_added(self):
        e = _fake_exception("NotFoundError", "missing")
        result = handle_openai_error(e)
        assert "response" not in result.raw_error  # type: ignore
        assert "request_id" not in result.raw_error  # type: ignore
        assert "body" not in result.raw_error  # type: ignore

    def test_timeout_error_with_connect_cause(self, httpx_mod):
        e = _fake_exception("APITimeoutError", "timed out")
        e.__cause__ = httpx_mod.ConnectTimeout("connect timeout")
        result = handle_openai_error(e)
        assert isinstance(result, TimeoutException)
        assert result.timeout_type == TimeoutType.CONNECT

    def test_timeout_error_with_read_cause(self, httpx_mod):
        e = _fake_exception("APITimeoutError", "timed out")
        e.__cause__ = httpx_mod.ReadTimeout("read timeout")
        result = handle_openai_error(e)
        assert result.timeout_type == TimeoutType.READ  # type: ignore

    def test_timeout_error_with_no_cause_has_none_timeout_type(self):
        e = _fake_exception("APITimeoutError", "timed out")
        result = handle_openai_error(e)
        assert isinstance(result, TimeoutException)
        assert result.timeout_type is None

    def test_timeout_error_with_unrelated_cause_has_none_timeout_type(self):
        e = _fake_exception("APITimeoutError", "timed out")
        e.__cause__ = ValueError("unrelated")
        result = handle_openai_error(e)
        assert result.timeout_type is None  # type: ignore

    def test_connection_error_has_none_status_by_default(self):
        e = _fake_exception("APIConnectionError", "no connection")
        result = handle_openai_error(e)
        assert isinstance(result, ServiceUnavailableException)
        assert result.status_code is None


class TestHandleAnthropicError:
    """Structurally identical dispatch to handle_openai_error — the map and
    the provider tag differ, everything else is shared logic."""

    def test_mapped_error_type_sets_anthropic_provider(self):
        e = _fake_exception("AuthenticationError", "bad key")
        result = handle_anthropic_error(e)
        assert isinstance(result, AuthenticationException)
        assert result.status_code == 401
        assert result.provider == ServiceProvider.ANTHROPIC

    def test_unmapped_error_type_falls_back(self):
        e = _fake_exception("TotallyUnknownError", "?")
        result = handle_anthropic_error(e)
        assert type(result) is LLMfyException
        assert result.provider == ServiceProvider.ANTHROPIC

    def test_timeout_with_write_cause(self, httpx_mod):
        e = _fake_exception("APITimeoutError", "timed out")
        e.__cause__ = httpx_mod.WriteTimeout("write timeout")
        result = handle_anthropic_error(e)
        assert isinstance(result, TimeoutException)
        assert result.timeout_type == TimeoutType.WRITE


class TestHandleGoogleError:
    def test_timeout_exception_maps_by_exact_type(self, httpx_mod):
        e = httpx_mod.ReadTimeout("read timeout")
        result = handle_google_error(e)
        assert isinstance(result, TimeoutException)
        assert result.timeout_type == TimeoutType.READ
        assert result.provider == ServiceProvider.GOOGLE

    def test_timeout_exception_status_code_stays_none(self, httpx_mod):
        # Unlike the bedrock/openai/anthropic timeout branches, google's
        # httpx/httpx2 TimeoutException branch never sets status_code.
        e = httpx_mod.PoolTimeout("pool timeout")
        result = handle_google_error(e)
        assert result.status_code is None

    def test_pool_timeout_type(self, httpx_mod):
        e = httpx_mod.PoolTimeout("pool timeout")
        result = handle_google_error(e)
        assert result.timeout_type == TimeoutType.POOL  # type: ignore

    def test_mapped_api_error_code(self):
        e = FakeGoogleAPIError(code=429, message="slow down")
        result = handle_google_error(e)
        assert isinstance(result, RateLimitException)
        assert result.status_code == 429
        assert result.message == "slow down"
        assert result.raw_error == {"code": 429, "message": "slow down"}

    def test_unmapped_api_error_code_falls_back(self):
        e = FakeGoogleAPIError(code=999, message="mystery")
        result = handle_google_error(e)
        assert type(result) is LLMfyException
        assert result.status_code == 999

    def test_details_included_when_present(self):
        e = FakeGoogleAPIError(code=400, message="bad", details={"field": "x"})
        result = handle_google_error(e)
        assert result.raw_error["details"] == {"field": "x"}  # type: ignore

    def test_details_absent_when_not_set(self):
        e = FakeGoogleAPIError(code=400, message="bad")
        result = handle_google_error(e)
        assert "details" not in result.raw_error  # type: ignore

    def test_completely_unrelated_exception_falls_back_to_generic(self):
        result = handle_google_error(ValueError("totally unrelated"))
        assert type(result) is LLMfyException
        assert result.provider == ServiceProvider.GOOGLE
        assert result.raw_error == {"error": "totally unrelated"}

    def test_model_error_and_not_found_reachable_only_via_bedrock_openai(self):
        # Documents an intentional taxonomy gap: ModelErrorException and
        # ModelNotFoundException-by-code-424/424 are Bedrock/OpenAI-specific;
        # Google's own map still supports ModelNotFoundException (404) though.
        e = FakeGoogleAPIError(code=404, message="model not found")
        result = handle_google_error(e)
        assert isinstance(result, ModelNotFoundException)
        assert not isinstance(result, ModelErrorException)
