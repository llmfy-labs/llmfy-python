"""Both retry layers end to end: real boto3/aioboto3 + LLMfy over local HTTP.

A local server stands in for Bedrock Converse and answers each request from a
script, so these tests check what the unit tests mock away: which failures
botocore retries under `SDKRetryConfig`, how llmfy maps the rest, and that
`LLMRetryPolicy` only retries what the SDK gave up on. Nothing reaches AWS.
"""

from __future__ import annotations

import json
import re
import threading
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest
from botocore.retries import standard

from llmfy.exception.llmfy_exception import (
    InvalidRequestException,
    ModelErrorException,
    RateLimitException,
    ServiceUnavailableException,
    TimeoutException,
    TimeoutType,
)
from llmfy.llmfy_core.llmfy import LLMfy
from llmfy.llmfy_core.llms.bedrock.converse.bedrock_converse_model import (
    BedrockConverseModel,
)
from llmfy.llmfy_core.retry import LLMRetryPolicy, SDKRetryConfig, retry_runner
from tests.llmfy_core.retry.conftest import FakeClock

READ_TIMEOUT = 0.5
"""Seconds the client waits for a reply; HANG stays silent longer than this."""

SDK_ATTEMPTS = 3


@dataclass(frozen=True)
class _Reply:
    status: int = 200
    error_code: str | None = None
    text: str = "halo"
    hang: bool = False


OK = _Reply()
EMPTY = _Reply(text="")
THROTTLE = _Reply(429, "ThrottlingException")
NOT_READY = _Reply(429, "ModelNotReadyException")
INTERNAL = _Reply(500, "InternalServerException")
UNAVAILABLE = _Reply(503, "ServiceUnavailableException")
HANG = _Reply(hang=True)
MODEL_TIMEOUT = _Reply(408, "ModelTimeoutException")
MODEL_ERROR = _Reply(424, "ModelErrorException")
VALIDATION = _Reply(400, "ValidationException")


class _MockBedrock:
    """Bedrock Converse endpoint that replays scripted replies, then answers OK."""

    def __init__(self) -> None:
        self.replies: deque[_Reply] = deque()
        self.sdk_attempts: list[int] = []
        self.release = threading.Event()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def script(self, *replies: _Reply) -> None:
        self.replies.extend(replies)

    def reset(self) -> None:
        self.replies.clear()
        self.sdk_attempts.clear()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                # botocore sends "attempt=2; max=3" (standard mode).
                attempt = re.search(r"attempt=(\d+)", self.headers.get("amz-sdk-request", ""))
                mock.sdk_attempts.append(int(attempt.group(1)) if attempt else 0)
                reply = mock.replies.popleft() if mock.replies else OK
                if reply.hang:
                    # The client gives up after READ_TIMEOUT and retries.
                    mock.release.wait(READ_TIMEOUT * 4)
                try:
                    self._send(reply)
                except OSError:
                    pass  # the client already hung up

            def _send(self, reply: _Reply) -> None:
                if reply.error_code:
                    body: dict[str, Any] = {"message": f"mock {reply.error_code}"}
                else:
                    body = {
                        "output": {
                            "message": {"role": "assistant", "content": [{"text": reply.text}]}
                        },
                        "stopReason": "end_turn",
                        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                        "metrics": {"latencyMs": 1},
                    }
                payload = json.dumps(body).encode()
                self.send_response(reply.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                if reply.error_code:
                    self.send_header("x-amzn-ErrorType", reply.error_code)
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        return Handler


@pytest.fixture(scope="module")
def _server() -> Iterator[_MockBedrock]:
    mock = _MockBedrock()
    thread = threading.Thread(target=mock.server.serve_forever, daemon=True)
    thread.start()
    yield mock
    mock.release.set()
    mock.server.shutdown()
    mock.server.server_close()


@pytest.fixture
def bedrock(_server: _MockBedrock, monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """Point Bedrock clients at the mock; policy sleeps are recorded, not slept."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "integration-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "integration-test")
    monkeypatch.setenv("AWS_BEDROCK_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", _server.url)
    monkeypatch.delenv("AWS_IGNORE_CONFIGURED_ENDPOINT_URLS", raising=False)
    # Keep both layers' backoff out of the test's run time.
    monkeypatch.setattr(standard.ExponentialBackoff, "delay_amount", lambda *_: 0)
    clock = FakeClock()
    monkeypatch.setattr(retry_runner, "_monotonic", clock.monotonic)
    monkeypatch.setattr(retry_runner, "_sleep", clock.sleep)
    monkeypatch.setattr(retry_runner, "_async_sleep", clock.async_sleep)
    _server.reset()
    yield SimpleNamespace(mock=_server, policy_sleeps=clock.sleeps)
    _server.reset()


def _llm(**policy: Any) -> LLMfy:
    model = BedrockConverseModel(
        model="integration-model",
        sdk_retry=SDKRetryConfig(max_attempts=SDK_ATTEMPTS, read_timeout=READ_TIMEOUT),
    )
    return LLMfy(model, retry=LLMRetryPolicy(**policy))


@pytest.mark.parametrize(
    "failure",
    [THROTTLE, NOT_READY, INTERNAL, UNAVAILABLE, HANG],
    ids=["throttling", "model-not-ready", "500", "503", "read-timeout"],
)
def test_sdk_retries_transport_failures(bedrock: SimpleNamespace, failure: _Reply):
    bedrock.mock.script(failure, failure, OK)

    response = _llm().invoke("hai")

    assert response.result.content == "halo"
    assert bedrock.mock.sdk_attempts == [1, 2, 3]
    assert bedrock.policy_sleeps == []


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        (THROTTLE, RateLimitException),
        (UNAVAILABLE, ServiceUnavailableException),
        (HANG, TimeoutException),
    ],
    ids=["throttling", "503", "read-timeout"],
)
def test_policy_does_not_repeat_what_the_sdk_gave_up_on(
    bedrock: SimpleNamespace, failure: _Reply, error: type[Exception]
):
    bedrock.mock.script(failure, failure, failure)

    with pytest.raises(error) as raised:
        _llm().invoke("hai")

    if isinstance(raised.value, TimeoutException):
        assert raised.value.timeout_type is TimeoutType.READ
    # Without a throttle_wait: one SDK call of three attempts, not multiplied.
    assert bedrock.mock.sdk_attempts == [1, 2, 3]
    assert bedrock.policy_sleeps == []


def test_policy_waits_out_throttling_that_outlasts_the_sdk(bedrock: SimpleNamespace):
    bedrock.mock.script(THROTTLE, THROTTLE, THROTTLE, THROTTLE, OK)

    response = _llm(throttle_wait=30).invoke("hai")

    assert response.result.content == "halo"
    # The SDK gives up after three attempts; the policy starts a fresh SDK call.
    assert bedrock.mock.sdk_attempts == [1, 2, 3, 1, 2]
    assert len(bedrock.policy_sleeps) == 1


def test_policy_gives_up_once_the_throttle_wait_runs_out(bedrock: SimpleNamespace):
    bedrock.mock.script(*[THROTTLE] * 60)

    with pytest.raises(RateLimitException):
        _llm(throttle_wait=10).invoke("hai")

    assert sum(bedrock.policy_sleeps) == pytest.approx(10)
    policy_calls = len(bedrock.policy_sleeps) + 1
    assert bedrock.mock.sdk_attempts == [1, 2, 3] * policy_calls


@pytest.mark.parametrize(
    "failure",
    [MODEL_TIMEOUT, MODEL_ERROR, EMPTY],
    ids=["model-timeout-408", "model-error-424", "empty-reply"],
)
def test_policy_retries_what_the_sdk_treats_as_final(bedrock: SimpleNamespace, failure: _Reply):
    bedrock.mock.script(failure, failure, OK)

    response = _llm().invoke("hai")

    assert response.result.content == "halo"
    # Every policy retry is a fresh SDK call that starts again at attempt 1.
    assert bedrock.mock.sdk_attempts == [1, 1, 1]
    assert len(bedrock.policy_sleeps) == 2


def test_policy_raises_once_its_attempts_run_out(bedrock: SimpleNamespace):
    bedrock.mock.script(MODEL_ERROR, MODEL_ERROR, MODEL_ERROR)

    with pytest.raises(ModelErrorException):
        _llm().invoke("hai")

    assert bedrock.mock.sdk_attempts == [1, 1, 1]
    assert len(bedrock.policy_sleeps) == 2


def test_layers_hand_over_within_one_call(bedrock: SimpleNamespace):
    # Throttled, then a model error the SDK passes up: the policy starts a new SDK call.
    bedrock.mock.script(THROTTLE, MODEL_ERROR, OK)

    response = _llm().invoke("hai")

    assert response.result.content == "halo"
    assert bedrock.mock.sdk_attempts == [1, 2, 1]
    assert len(bedrock.policy_sleeps) == 1


def test_invalid_requests_are_not_retried(bedrock: SimpleNamespace):
    bedrock.mock.script(VALIDATION)

    with pytest.raises(InvalidRequestException):
        _llm().invoke("hai")

    assert bedrock.mock.sdk_attempts == [1]
    assert bedrock.policy_sleeps == []


async def test_async_path_uses_the_same_layers(bedrock: SimpleNamespace):
    pytest.importorskip("aioboto3")
    bedrock.mock.script(THROTTLE, THROTTLE, THROTTLE, MODEL_ERROR, OK)

    response = await _llm(throttle_wait=30).ainvoke("hai")

    assert response.result.content == "halo"
    assert bedrock.mock.sdk_attempts == [1, 2, 3, 1, 1]
    assert len(bedrock.policy_sleeps) == 2
