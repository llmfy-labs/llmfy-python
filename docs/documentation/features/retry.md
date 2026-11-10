# Retry

LLM calls fail in two different ways, so LLMfy retries in two layers:

| Layer | Configured with | Retries | Typical time |
|---|---|---|---|
| **SDK** (boto3, openai, anthropic, google-genai) | `SDKRetryConfig` on the model | throttling, 5xx, dropped connections, read timeouts | seconds |
| **LLMfy** | `LLMRetryPolicy` on `LLMfy` | model errors, model timeouts, empty replies; capacity errors that outlast the SDK (opt-in) | up to `throttle_wait` |

Both layers are off unless you configure them, so existing code behaves exactly as before.

---

## Quick start

```python linenums="1"
from llmfy import BedrockConverseModel, LLMfy, LLMRetryPolicy, SDKRetryConfig

llm = BedrockConverseModel(
    model="amazon.nova-pro-v1:0",
    # SDK layer: 3 quick attempts, give up on a silent connection after 60s.
    sdk_retry=SDKRetryConfig(max_attempts=3, read_timeout=60),
)

chat = LLMfy(
    llm,
    system_message="You are a helpful assistant.",
    # LLMfy layer: retry model errors and empty replies up to 3 attempts,
    # and keep retrying throttling for up to 30 seconds.
    retry=LLMRetryPolicy(max_attempts=3, throttle_wait=30),
)

response = chat.invoke("Hello!")
```

---

## `LLMRetryPolicy`

The policy wraps **each model round trip**, never a whole tool-calling loop. In `invoke_with_tools`/`chat_with_tools`, a tool that already ran is never run again: only the failed request to the model is repeated.

The policy belongs to the `LLMfy` instance, not the model. Two `LLMfy` instances can share one model object and still use different policies:

```python linenums="1"
llm = BedrockConverseModel(model="amazon.nova-pro-v1:0")

# A person is waiting: give up quickly.
chat = LLMfy(llm, retry=LLMRetryPolicy(throttle_wait=30))
# Nobody is waiting: ride out a spent per-minute quota.
background = LLMfy(llm, retry=LLMRetryPolicy(throttle_wait=120))
```

### What is retried

| Failure | Default handling |
|---|---|
| `ModelErrorException` (Bedrock 424) | retried, counts against `max_attempts` |
| `TimeoutException` with `timeout_type=TimeoutType.MODEL` (Bedrock 408) | retried, counts against `max_attempts` |
| Empty reply (no non-blank text and no tool calls) | retried, counts against `max_attempts` |
| `RateLimitException`, `ServiceUnavailableException`, status 429/503/529 | retried only while `throttle_wait` lasts; does **not** use up `max_attempts` |
| Network timeouts (`READ`, `CONNECT`, …) | not retried: the SDK already did, and repeating them multiplies long hangs |
| Invalid request, authentication, content filter, anything else | not retried |

### Options

| Option | Default | Meaning |
|---|---|---|
| `max_attempts` | `3` | Total attempts for retryable errors and empty replies, including the first call. `1` disables them. |
| `base_delay` | `1.0` | Seconds before the second attempt; doubles after each retry. |
| `max_delay` | `8.0` | Cap for one delay. |
| `throttle_wait` | `0` | Seconds to keep retrying capacity errors, counted from the first one. `0` gives up when the SDK does. |
| `throttle_base_delay` | `2.0` | Seconds before the first capacity retry; doubles after each one. |
| `throttle_max_delay` | `15.0` | Cap for one capacity delay. The last delay is also clamped to what remains of `throttle_wait`. |
| `retry_on_empty` | `True` | Retry empty replies. |
| `on_empty_exhausted` | `"return"` | When every attempt came back empty: `"return"` the last empty reply or `"raise"` `EmptyResponseException`. |
| `retry_stream` | `True` | Retry `*_stream` calls that fail before their first chunk. |
| `is_retryable` | `is_retryable_llm_error` | `Callable[[BaseException], bool]` for errors counted against `max_attempts`. |
| `is_capacity_error` | `is_capacity_error` | `Callable[[BaseException], bool]` for errors governed by `throttle_wait`. |
| `on_retry` | `None` | `Callable[[RetryEvent], None]` called before every sleep. |

Every delay gets random jitter (50–100% of its value), so callers that failed together don't retry in lockstep.

!!! note "Daily quotas"
    Bedrock reports a spent **daily** token quota with the same `ThrottlingException` as a per-minute one. With a `throttle_wait`, such a call waits the full wait and then fails anyway.

### Streaming

`invoke_stream`, `chat_stream`, `ainvoke_stream` and `achat_stream` are retried only **until the first chunk reaches you**. After that, an error is raised as-is, because a retry would repeat text you already received. Empty replies are not retried for streams.

### Sync vs. async

Sync methods sleep with `time.sleep`, which blocks the calling thread for the whole retry. Async methods (`ainvoke`, `achat`, …) sleep with `asyncio.sleep` and don't block the event loop.

### Logging and metrics

Each retry is logged at `WARNING` level on the `llmfy.llmfy_core.retry.retry_runner` logger:

```shell
[LLM Retry] amazon.nova-pro-v1:0 attempt 1/3 failed: ModelErrorException(...); retrying in 0.8s
[LLM Retry] amazon.nova-pro-v1:0 out of capacity: RateLimitException(...); retry 1 in 1.6s (30s of 30s wait left)
```

For metrics, pass `on_retry`. It receives a `RetryEvent` with `model`, `reason` (`"error"`, `"empty"` or `"capacity"`), `attempt`, `max_attempts`, `delay`, `error`, `wait_remaining` and `streaming`:

```python linenums="1"
from llmfy import LLMRetryPolicy, RetryEvent

def record_retry(event: RetryEvent) -> None:
    metrics.increment("llm.retry", tags={"model": event.model, "reason": event.reason})

policy = LLMRetryPolicy(throttle_wait=30, on_retry=record_retry)
```

Exceptions raised by `on_retry` propagate to the caller.

### Custom classifiers

```python linenums="1"
from llmfy import InvalidRequestException, LLMRetryPolicy
from llmfy.llmfy_core.retry import is_retryable_llm_error

def retry_flaky_proxy(error: BaseException) -> bool:
    if isinstance(error, InvalidRequestException) and error.status_code == 422:
        return True
    return is_retryable_llm_error(error)

policy = LLMRetryPolicy(is_retryable=retry_flaky_proxy)
```

---

## `SDKRetryConfig`

Pass it to any model as `sdk_retry=`. Fields left as `None` keep the SDK's own default.

| Field | Bedrock (botocore) | OpenAI / Anthropic | Google |
|---|---|---|---|
| `max_attempts` | `"standard"` retry mode, `total_max_attempts` | `max_retries = max_attempts - 1` | `HttpRetryOptions(attempts=...)` |
| `read_timeout` | `read_timeout` | `Timeout(read=...)` | request `timeout` |
| `connect_timeout` | `connect_timeout` | `Timeout(connect=...)` | ignored |

The SDK defaults this replaces: botocore uses `"legacy"` mode with 5 attempts and 60s timeouts; OpenAI and Anthropic retry twice with a 600s read timeout; Google does not retry.

Bedrock also accepts `client_config`, a botocore `Config` for anything else (for example a wider connection pool). It's used for both the sync and the aioboto3 client, and `sdk_retry` overrides its retry and timeout fields:

```python linenums="1"
from botocore.config import Config

llm = BedrockConverseModel(
    model="amazon.nova-pro-v1:0",
    sdk_retry=SDKRetryConfig(max_attempts=3, read_timeout=60),
    client_config=Config(max_pool_connections=32),
)
```

!!! warning "The layers multiply"
    Every `LLMRetryPolicy` attempt runs up to `SDKRetryConfig.max_attempts` SDK requests, and a hung connection can take `max_attempts × read_timeout` inside one attempt. Keep the SDK layer short (a few attempts, a bounded `read_timeout`) and let `throttle_wait` decide how long a call may take overall.

---

## Retry in FlowEngine

FlowEngine's `RetryPolicy` (on a `Node`) is a different, coarser tool: it re-runs the **whole node**, including any tools the node already called. For LLM failures, prefer an `LLMRetryPolicy` on the `LLMfy` the node uses, and keep the node `RetryPolicy` for failures of the node's own code.
