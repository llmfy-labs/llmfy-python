# Exception Handler

LLMfy maps provider-specific errors into a unified exception hierarchy so you can handle errors consistently regardless of which provider you use.

## Base Exception

All LLMfy exceptions inherit from `LLMfyException`:

```python
from llmfy import LLMfyException
```

`LLMfyException` exposes four attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable error message |
| `status_code` | `int \| None` | HTTP status code from the provider |
| `provider` | `str \| None` | Provider name (e.g. `"anthropic"`, `"openai"`, `"bedrock"`, `"google"`) |
| `raw_error` | `Any \| None` | The original exception from the provider SDK |

## Exception Types

| Exception | Description |
|-----------|-------------|
| `LLMfyException` | Base class for all LLMfy errors |
| `RateLimitException` | Rate limit exceeded (HTTP 429) |
| `QuotaExceededException` | Usage/quota limit exceeded |
| `TimeoutException` | Request timed out (HTTP 408) — see [Timeout Types](#timeout-types) |
| `InvalidRequestException` | Invalid request parameters (HTTP 400/422) |
| `AuthenticationException` | Authentication failed (HTTP 401/403) |
| `PermissionDeniedException` | Permission denied (HTTP 403) |
| `ModelNotFoundException` | Model not found or unavailable (HTTP 404) |
| `ServiceUnavailableException` | Service temporarily unavailable (HTTP 500/503) |
| `ContentFilterException` | Content blocked by safety filters |
| `ModelErrorException` | Model processing error (HTTP 424) |
| `EmptyResponseException` | Every attempt returned an empty reply; raised only with `LLMRetryPolicy(on_empty_exhausted="raise")` — see [Retry](../features/retry.md) |

## Basic Usage

Catch `LLMfyException` to handle any LLMfy error:

```python linenums="1"
from llmfy import LLMfy, LLMfyException

try:
    response = agent.invoke("Hello")
    print(response.result.content)
except LLMfyException as e:
    print(f"Error: {e.message}")
    print(f"Status Code: {e.status_code}")
    print(f"Provider: {e.provider}")
```

## Handling Specific Exceptions

Import specific exception types to handle them differently:

```python linenums="1"
from llmfy import (
    LLMfyException,
    RateLimitException,
    TimeoutException,
    AuthenticationException,
    ServiceUnavailableException,
)

try:
    response = agent.invoke("Hello")
    print(response.result.content)
except RateLimitException as e:
    print(f"Rate limited (HTTP {e.status_code}) — implement backoff and retry.")
except TimeoutException as e:
    print(f"Request timed out — retry with a shorter prompt or increase timeout.")
except AuthenticationException as e:
    print(f"Authentication failed — check your API key or credentials.")
except ServiceUnavailableException as e:
    print(f"Service unavailable (HTTP {e.status_code}) — retry later.")
except LLMfyException as e:
    print(f"Unexpected error: {e.message}")
```

## Timeout Types

`TimeoutException` carries an additional `timeout_type` attribute (`TimeoutType` enum) that identifies the exact cause of the timeout. This lets you apply different retry strategies depending on where the failure occurred.

```python
from llmfy import TimeoutException, TimeoutType
```

| `TimeoutType` | Value | Description |
|---------------|-------|-------------|
| `TimeoutType.CONNECT` | `"connect"` | Could not establish a connection to the endpoint |
| `TimeoutType.READ` | `"read"` | Connected but timed out waiting for response bytes |
| `TimeoutType.WRITE` | `"write"` | Timed out sending the request body |
| `TimeoutType.POOL` | `"pool"` | Timed out waiting for a connection slot from the pool |
| `TimeoutType.MODEL` | `"model"` | Model-side processing timeout (Bedrock `ModelTimeoutException`) |
| `None` | — | Type could not be determined |

### Per-provider timeout coverage

| Provider | Trigger | `timeout_type` |
|----------|---------|----------------|
| **Anthropic** | `APITimeoutError` (inspects `__cause__`) | `CONNECT` / `READ` / `WRITE` / `POOL` / `None` |
| **Bedrock** | `botocore.ReadTimeoutError` | `CONNECT` |
| **Bedrock** | `botocore.ConnectTimeoutError` | `READ` |
| **Bedrock** | `ClientError: ModelTimeoutException` | `MODEL` |
| **OpenAI** | `APITimeoutError` (inspects `__cause__`) | `CONNECT` / `READ` / `WRITE` / `POOL` / `None` |
| **Google** | `httpx2.ConnectTimeout` / `httpx.ConnectTimeout` | `CONNECT` |
| **Google** | `httpx2.ReadTimeout` / `httpx.ReadTimeout` | `READ` |
| **Google** | `httpx2.WriteTimeout` / `httpx.WriteTimeout` | `WRITE` |
| **Google** | `httpx2.PoolTimeout` / `httpx.PoolTimeout` | `POOL` |

For Anthropic and OpenAI, `timeout_type` comes from the type of the SDK error's `__cause__`, which is one of the same four transport timeouts. Current provider SDK releases use [`httpx2`](https://github.com/pydantic/httpx2), a fork of `httpx` under a separate import name, and older releases use `httpx`. LLMfy recognizes the timeout classes from whichever of the two packages is installed, and from both when both are.

### Example

```python linenums="1"
from llmfy import TimeoutException, TimeoutType

try:
    response = agent.invoke("Hello")
    print(response.result.content)
except TimeoutException as e:
    if e.timeout_type == TimeoutType.CONNECT:
        print("Endpoint unreachable — check network or retry later.")
    elif e.timeout_type == TimeoutType.MODEL:
        print("Model overloaded — reduce payload size or wait longer before retrying.")
    elif e.timeout_type in (TimeoutType.READ, TimeoutType.WRITE):
        print("Transport timeout — retry with exponential backoff.")
    elif e.timeout_type == TimeoutType.POOL:
        print("Connection pool exhausted — reduce concurrency or increase pool size.")
    else:
        print(f"Request timed out: {e.message}")
```

## Provider Error Mapping

LLMfy automatically maps provider-specific errors to the corresponding exception type.

### Anthropic

The `anthropic` SDK's exception hierarchy is built on `httpx2` (`httpx` in older releases) and structurally analogous to OpenAI's, so the mapping is identical in shape:

| Provider Error | LLMfy Exception | Status Code |
|----------------|----------------|-------------|
| `RateLimitError` | `RateLimitException` | 429 |
| `APITimeoutError` | `TimeoutException` | 408 |
| `APIConnectionError` | `ServiceUnavailableException` | — |
| `AuthenticationError` | `AuthenticationException` | 401 |
| `PermissionDeniedError` | `PermissionDeniedException` | 403 |
| `BadRequestError` | `InvalidRequestException` | 400 |
| `NotFoundError` | `ModelNotFoundException` | 404 |
| `UnprocessableEntityError` | `InvalidRequestException` | 422 |
| `InternalServerError` | `ServiceUnavailableException` | 500 |

### OpenAI

| Provider Error | LLMfy Exception | Status Code |
|----------------|----------------|-------------|
| `RateLimitError` | `RateLimitException` | 429 |
| `APITimeoutError` | `TimeoutException` | 408 |
| `APIConnectionError` | `ServiceUnavailableException` | — |
| `AuthenticationError` | `AuthenticationException` | 401 |
| `PermissionDeniedError` | `PermissionDeniedException` | 403 |
| `BadRequestError` | `InvalidRequestException` | 400 |
| `NotFoundError` | `ModelNotFoundException` | 404 |
| `UnprocessableEntityError` | `InvalidRequestException` | 422 |
| `InternalServerError` | `ServiceUnavailableException` | 500 |

### AWS Bedrock

| Provider Error | LLMfy Exception | Status Code |
|----------------|----------------|-------------|
| `ThrottlingException` | `RateLimitException` | 429 |
| `ModelTimeoutException` | `TimeoutException` | 408 |
| `ModelNotReadyException` | `ServiceUnavailableException` | 429 |
| `ValidationException` | `InvalidRequestException` | 400 |
| `AccessDeniedException` | `AuthenticationException` | 403 |
| `ResourceNotFoundException` | `ModelNotFoundException` | 404 |
| `ServiceUnavailableException` | `ServiceUnavailableException` | 503 |
| `InternalServerException` | `ServiceUnavailableException` | 500 |
| `ModelErrorException` | `ModelErrorException` | 424 |

### Google AI

| HTTP Status | LLMfy Exception |
|-------------|----------------|
| 400 | `InvalidRequestException` |
| 401 | `AuthenticationException` |
| 403 | `PermissionDeniedException` |
| 404 | `ModelNotFoundException` |
| 408 | `TimeoutException` |
| 429 | `RateLimitException` |
| 500 | `ServiceUnavailableException` |
| 503 | `ServiceUnavailableException` |
