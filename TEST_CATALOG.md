# LLMfy behavior checklist (for porting to other language SDKs)

This is a plain-language checklist of every observable behavior the Python
`llmfy` package's test suite (`tests/`) verifies. It exists so a port to
another language doesn't require reading Python test code (fixtures,
`unittest.mock`, `monkeypatch`, pytest idioms) to know what to reproduce —
read this instead, then consult the referenced test file only when a bullet
needs more precision than a sentence can give.

**How to use this for a port**: copy this file into the new SDK's repo and
check items off (`- [ ]` → `- [x]`) as its own test suite proves each
behavior. Every box here is already checked for the Python implementation —
they're all backed by a passing test in `tests/` as of this writing. A box
you can't check means either a genuine behavior gap in the port, or a
deliberate, documented difference — which should be written down, not left
silent.

**Scope**: everything below mirrors `tests/`, which covers all of `llmfy/`,
including `llmfy/flow_engine/` (`tests/flow_engine/`, mirroring the
package layout).

Each section names its source test file(s) so behavior and proof stay linked.

---

## Public API contract
*`tests/test_public_api.py`*

- [x] Every name in `llmfy.__all__` is actually importable from the top-level package.
- [x] `__all__` has no duplicate entries.
- [x] Core chat API is exported: `LLMfy`, `Message`, `Role`, `Tool`, `ToolRegistry`, `AIResponse`, `GenerationResponse`.
- [x] Every provider's model + config class is exported (all 5 backends).
- [x] The full exception hierarchy is exported, and every exported exception class is a subclass of `LLMfyException`.
- [x] Chunking/text-preprocessing utilities are exported: `chunk_text`, `chunk_markdown_by_header`, `clean_text_for_embedding`.
- [x] `__version__` is exported and is a string.

## Supply-chain / optional-dependency isolation
*`tests/test_dependency_isolation.py`*

- [x] With **zero** provider SDKs installed (`openai`, `boto3`/`aioboto3`, `anthropic`, `google-genai` all blocked), `import llmfy` still succeeds.
- [x] Provider-agnostic features still work with no SDKs installed: constructing `Message`, using `@Tool`, calling `chunk_text`, raising/catching `LLMfyException`, constructing `Content`.
- [x] Instantiating `OpenAIChatModel` without `openai` installed raises a clean `LLMfyException` naming the missing package and the install extra — never a raw `ModuleNotFoundError`.
- [x] Same clean-failure guarantee for `AnthropicMessagesModel` (missing `anthropic`), `BedrockConverseModel` (missing `boto3`), `GoogleAIGenerateModel` (missing `google-genai`).

## Exceptions
*`tests/exception/`*

**`LLMfyException` hierarchy** (`test_llmfy_exception.py`)
- [x] Base exception constructs with just a message; `status_code`/`raw_error`/`provider` all default to `None`.
- [x] `str(exception) == message` (standard `Exception` behavior via `super().__init__(message)`).
- [x] `repr()` format is exactly `ClassName(message='...', status_code=..., provider='...')` — note `raw_error` is deliberately excluded from `repr()` (could carry sensitive data).
- [x] Every leaf subclass (`RateLimitException`, `QuotaExceededException`, `InvalidRequestException`, `AuthenticationException`, `PermissionDeniedException`, `ModelNotFoundException`, `ServiceUnavailableException`, `ContentFilterException`, `ModelErrorException`) is a plain subclass with no added fields, and `repr()` uses the concrete class name.
- [x] `TimeoutException` adds one extra field, `timeout_type` (an enum: `CONNECT`, `READ`, `WRITE`, `POOL`, `MODEL`), defaulting to `None`. `TimeoutException` does **not** override `repr()`, so `timeout_type` never appears in it.

**Error-code → exception-class maps** (`test_exception_mapper.py`) — pin the exact shape, since the error handlers depend on it:
- [x] Bedrock: 9 entries keyed by AWS error-code string, each mapping to `(ExceptionClass, default_http_status)`. `ThrottlingException`→429 rate-limit, `ModelTimeoutException`→408 timeout, `ValidationException`→400 invalid-request, `AccessDeniedException`→403 auth, `ResourceNotFoundException`→404 model-not-found.
- [x] OpenAI and Anthropic share the same 9 SDK-exception-class-name keys and the same shape (both SDKs use matching class names): `RateLimitError`→429, `APITimeoutError`→408 timeout, `APIConnectionError`→`(ServiceUnavailable, None)` (no default status — a network failure has no HTTP response), `AuthenticationError`→401, `PermissionDeniedError`→403, `BadRequestError`→400, `NotFoundError`→404, `UnprocessableEntityError`→422, `InternalServerError`→500.
- [x] Google is keyed directly by **HTTP status int** (not a class name), values are plain exception classes (no status tuple, since the key already is the status): 400, 401, 403, 404, 408, 429, 500, 503 all mapped; 403 → `PermissionDeniedException`.

**Error handlers** (`test_exception_handler.py`) — tested against real SDK exception instances (`botocore`, `httpx`, `google-genai`), not mocks, for the ones that dispatch via `isinstance`:
- [x] Bedrock: `ReadTimeoutError`/`ConnectTimeoutError` (real botocore exceptions) map to `TimeoutException` with `timeout_type=READ`/`CONNECT` respectively — checked *before* the generic `ClientError` handling.
- [x] A non-`ClientError`, non-timeout exception becomes a plain `LLMfyException` (status code stays `None`).
- [x] A mapped `ClientError` error code produces the mapped exception class; an explicit non-zero `HTTPStatusCode` from the response always wins over the map's default status; a missing `HTTPStatusCode` falls back to the map default; an unknown error code falls back to a plain `LLMfyException`.
- [x] A missing `Message` key in the error response falls back to `str(exception)`.
- [x] `ModelTimeoutException` (a *ClientError code*, not a raw botocore timeout type) still resolves to `TimeoutException` with `timeout_type=MODEL`.
- [x] OpenAI/Anthropic (identical dispatch logic, different provider tag): dispatch is purely by `type(exception).__name__` string match plus `hasattr` checks — never `isinstance` against the real SDK — so an unknown exception-class name always falls back to plain `LLMfyException`.
- [x] `response`/`request_id`/`body` attributes are copied into `raw_error` only when present on the source exception (never invented).
- [x] An explicit non-falsy `status_code` on the source exception wins over the map's default; a falsy/absent one falls back to the map default.
- [x] Timeout-type resolution for OpenAI/Anthropic inspects `exception.__cause__`'s exact type against `httpx.ConnectTimeout`/`ReadTimeout`/`WriteTimeout`/`PoolTimeout` — an unrelated or missing cause yields `timeout_type=None`.
- [x] Google: `httpx.TimeoutException` subclasses map to `TimeoutException` (status_code stays `None` here — an asymmetry versus the other three providers, which do set one). A real `google.genai.errors.APIError` maps by its `.code` (HTTP status); an unmapped code falls back to plain `LLMfyException`; `.details` is copied into `raw_error` only when present. Any other exception type (not a timeout, not an `APIError`) becomes a generic `LLMfyException` with `raw_error={"error": str(e)}`.

**Package exports** (`test_exception_package_exports.py`)
- [x] Every name in `llmfy.exception.__all__` is importable and is a subclass of `LLMfyException`.
- [x] `TimeoutType` is intentionally **not** re-exported at the `llmfy.exception` package level (only from `llmfy_exception` directly).

## Messages
*`tests/llmfy_core/messages/`*

**`Message`** (`test_message.py`)
- [x] Valid role/field combinations construct without error (user+content, system+content, assistant+tool_calls, tool+tool_call_id+tool_results).
- [x] Four cross-field invariants, checked in this exact order (a message violating more than one always fails on the first): `tool_results` set on a non-`tool` role → error; `tool` role with no (or empty-list) `tool_results` → error; `tool_call_id` set on a non-`tool` role → error; `tool_calls` set on a non-`assistant` role → error.
- [x] These validation errors are raised via the framework's model-validation mechanism (a pydantic `ValidationError` in the Python implementation, wrapping the original message text) — a port should decide and document its own equivalent (exception type or `Result`/`Either`).
- [x] `id` defaults to a fresh unique value per instance; `timestamp` defaults to an ISO-8601 string.
- [x] Unknown/extra fields on construction are rejected (strict schema).
- [x] Role accepts a plain string and coerces it to the enum; an invalid role string is rejected.

**`MessageBufferBuilder`** (`test_message_buffer_builder.py`) — in-memory, per-request chat history:
- [x] `add_system_message` always inserts at the **front** of history, not the end.
- [x] Calling `add_system_message` twice keeps **both** (no dedup/replace) — most recent ends up frontmost.
- [x] `add_user_message` appends with the given id.
- [x] `add_assistant_message` mutates each given `ToolCall`'s `request_call_id` **in place** to the new message's id — a documented side effect on the caller's own objects, not a copy.
- [x] `add_tool_message` delegates to the backend-specific formatter's own tool-result logic (see per-provider formatter sections below); an unsupported/unregistered backend raises a clean error.
- [x] `get_messages(backend)` formats every message in history, in order, for that backend; an unsupported backend raises a clean error.
- [x] Formatting is **cached per message id per backend** — calling `get_messages` repeatedly on an unchanged history does not re-format already-formatted messages; only genuinely new messages get formatted.
- [x] The cache evicts entries for messages no longer in history (e.g. after `clear()`), so it can't grow unbounded across many calls on one instance.
- [x] `get_instance_messages()` returns a **live reference** to the internal list, not a defensive copy — mutating the returned list mutates internal state.
- [x] `clear()` empties history but keeps the most-recently-inserted system message (if any) — this is "reset conversation, keep the current system prompt," and only ever keeps **one** system message even if multiple were added.

**`Content`** (`test_content.py`)
- [x] `type` defaults to `TEXT`. `value` is required (no default) and accepts either a string or raw bytes.
- [x] Optional fields (`filename`, `format`, `use_s3`, `bucket_owner`) all have sensible defaults (`None`/`False`).
- [x] Unknown/extra fields rejected (strict schema).
- [x] **No** provider-specific format/content validation happens at this level (e.g. an invalid image format string is accepted here) — that validation is each provider formatter's job, not this shared data class's.

**`ContentType` / `Role` enums** (`test_content_type.py`, `test_role.py`)
- [x] Exact string values for every member (`ContentType`: text/image/document/video; `Role`: system/user/assistant/tool).
- [x] String-enum members compare equal to their plain string value.
- [x] Construct-from-string works for a valid value; an invalid value is rejected.

**`ToolCall`** (`test_tool_call.py`)
- [x] All 4 fields (`tool_call_id`, `request_call_id`, `name`, `arguments`) are required — missing any one is rejected.
- [x] Unlike `Message`/`Content`, **unknown extra fields are silently ignored** here, not rejected — a deliberate contrast worth preserving or explicitly deciding against in a port.
- [x] `arguments` accepts arbitrarily nested structures (dict of dict of list, etc.).

## Tools
*`tests/llmfy_core/tools/`*

**Docstring parameter-description extraction** (`test_function_param_desc_extractor.py`)
- [x] Supports three docstring styles, tried in this exact priority order: Google-style (`name (type): description`), reST-style (`:param name: description`), Sphinx-with-type-style (`:param type name: description`).
- [x] Multi-line descriptions are collapsed to a single line (internal whitespace/newlines normalized to single spaces).
- [x] Description extraction correctly stops before the next parameter or before `Returns:`/`Raises:`/etc. sections.
- [x] No match (empty docstring, param not present, garbage input) returns an empty string — never raises.
- [x] A parameter name that's a substring of another name in the docstring (e.g. `loc` vs. `location`) does **not** false-match.
- [x] **Security**: parameter names are matched literally, not as regex syntax — a name containing regex metacharacters (`.`, `*`, `(`, `|`) must not corrupt the match, widen it unexpectedly, or crash the parser.

**Function metadata extraction** (`test_function_parser.py`)
- [x] Extracts function name, a short description (text before the first `Args:`/`:param`/etc. section — falls back to the whole docstring if no such section exists), the raw parameter signature, resolved type hints, and the raw docstring.
- [x] No docstring → empty description and empty docstring string (never `None`, never raises).
- [x] Default values on parameters are reflected faithfully in the extracted signature.
- [x] Extracting metadata from a bound method includes `self` in the parameter list (callers are expected to skip it, which every formatter does — see below).

**Type mapping** (`test_function_type_mapping.py`)
- [x] Fixed mapping from Python's `int`/`float`/`str`/`bool`/`list`/`dict`/`NoneType` to JSON-Schema-style type names (`integer`/`number`/`string`/`boolean`/`array`/`object`/`null`). Any other Python type falls back to `"string"` at the call site (the mapping itself has no catch-all entry).

**`@Tool` decorator** (`test_tool.py`)
- [x] Marks the decorated function (default `strict=True`, or the value passed to `Tool(strict=...)`) without wrapping it — the function is returned unchanged and remains directly callable with its original behavior.
- [x] The decorator does **not** inspect or cache the docstring at decoration time — reassigning `__doc__` after decoration is picked up correctly when the tool definition is later built.
- [x] Building a tool definition dispatches to the registered formatter for every one of the 5 real backends without error; an unregistered/missing backend raises a clean error.

**Tool registry** (`test_tool_registry.py`)
- [x] Registering a function not decorated with `@Tool` raises a clean error (checked via a simple attribute-presence marker, not a type check).
- [x] Registering two functions with the same `__name__` silently keeps only the later one (last-write-wins on both the callable and its schema — no error).
- [x] `get_tool_definitions()` returns definitions in registration order (post-dedup).
- [x] `execute_tool(name, arguments)` calls the registered function via dict lookup + keyword-argument spread — never `eval`/dynamic-attribute-lookup on an arbitrary object.
- [x] **Security**: calling with an unregistered name — including deliberately hostile names like `__import__`, `os.system`, or a path-traversal-shaped string — always raises a clean "tool not found" error and never executes anything.
- [x] A tool call with missing/wrong arguments raises a plain `TypeError` from the underlying function, unwrapped — the registry does not catch or reinterpret it.

## Usage tracking
*`tests/llmfy_core/usage/`*

**Pricing-table validation** (`test_llmfy_usage.py`)
- [x] OpenAI/Anthropic pricing: every model entry must be a dict whose values are all numeric (int/float); anything else is rejected.
- [x] Bedrock pricing: only the first two levels of dict nesting are checked (model → region → dict) — the innermost leaf values (price numbers) are **not** validated here, unlike OpenAI/Anthropic; a malformed leaf (missing `input`/`output` key) passes this check and fails later, downstream, when building the internal pricing model.
- [x] Google pricing supports 4 combinable shapes: flat numeric, per-content-type (a dict with a required `default` key), tiered (`threshold`+`input_high`+`output_high`, which must **all** be present together — partial tier config is rejected), and tiered+typed combined. Missing `input`/`output` keys altogether is rejected.
- [x] Passing an explicitly **empty** pricing dict (`{}`) for any provider skips validation (an empty dict is "falsy") **and** falls back to that provider's bundled default pricing table — it does *not* mean "use zero pricing." This footgun is worth flagging explicitly in a port rather than silently replicating.
- [x] Passing nothing (`None`, the default) uses the bundled default pricing table for every provider.

**Usage-update dispatch** (`test_llmfy_usage.py`)
- [x] `update()` never raises for an unrecognized combination of service-type/backend/provider — it's a silent no-op (all counters stay at zero). This applies to: LLM type with no/unknown backend, embedding type with no/unknown provider (including `ANTHROPIC`, which has no embedding update path by design), and a completely unrecognized service type.
- [x] Usage payloads are accepted either as a plain dict or as an arbitrary object exposing the same fields as attributes (`vars(obj)` is used internally) — both must produce identical accounting.

**Per-provider cost math** (one section per backend in `test_llmfy_usage.py`) — each of these is a concrete, checkable numeric contract a port must reproduce exactly:
- [x] OpenAI Chat/Responses: input/output token counts and cost use the standard `tokens / token_unit * price` formula; **cache-read and cache-write tokens are subsets of the input-token count** and are billed at their own (possibly custom) rate, subtracted from the "regular" input portion first so nothing is double-counted. Cache-read rate defaults to 50% of the input price if not set; cache-write defaults to 0 (free) if not set. Responses API uses different raw field names (`input_tokens`/`output_tokens` vs. Chat's `prompt_tokens`/`completion_tokens`) but shares the *same* pricing table (pricing is per-model, not per-API-variant).
- [x] Bedrock: pricing is looked up by **model AND region** (an environment variable supplies the region) — a model present in the pricing table but missing the specific region raises a raw `KeyError` (this is *not* treated as "model not found," which only warns). Cache-read defaults to 10% of the input rate, cache-write defaults to 125% of the input rate, both additive to the base input cost (not subtracted from it).
- [x] Anthropic: same additive cache-rate defaults and math as Bedrock (10% read / 125% write), but pricing has **no region dimension** — a flat per-model lookup.
- [x] Google: **the only provider with a fundamentally different cache-discount formula.** The full (non-discounted) input cost is computed first from the raw token count, then a "savings" amount (`normal_price_for_cached_tokens − actual_cached_price`) is *subtracted* from the total — mathematically equivalent to the other providers' "subtract cached tokens before pricing" approach only when the discount ratio matches the assumed default (25% of the input rate, i.e. 75% off), but computed via a different code path. A tiered (threshold-based) rate applies when total input tokens exceed a configured threshold, falling back to the normal rate if the high-tier rate is absent from the pricing entry even when the threshold is exceeded. Per-content-type pricing (text/image/video/audio each billed at a different rate, defaulting to a `default` rate for any type without its own entry) is only used when the usage payload actually reports a type breakdown; otherwise a flat `default` rate applies even for typed pricing entries.
- [x] Embedding updates (OpenAI, Bedrock, Google) always force `output_tokens = 0` and never touch cache accounting (embeddings have no output tokens or caching in this SDK's model). Bedrock's embedding usage reads its token count from an unusual hyphenated key (a raw AWS response field name), unlike every other usage payload's key naming.
- [x] Every "model not found in pricing table" case (LLM and embedding, every provider) emits a **warning** (not an error), still records the request in the running totals with a **zero** cost contribution, and still appends a detail entry (with `input_price`/`output_price` left `None` so it's distinguishable from a genuinely free request).

**Reporting / reset** (`test_llmfy_usage.py`)
- [x] `to_dict()` produces a stable nested summary shape (`total_request`, `tokens.*`, `costs.*` including a human-formatted trimmed-decimal cost string, `cache.*`, `details`).
- [x] The human-readable `repr()` omits its "Cache:" section entirely when no cache activity has occurred, and omits the per-request "backend:" line for embedding entries (which have no backend, only a provider).
- [x] `reset()` zeroes every running counter and clears history lists, but **does not** touch the loaded pricing tables — pricing persists across a reset.

**Usage-tracker context manager** (`test_usage_tracker.py`)
- [x] `with llmfy_usage_tracker() as usage:` yields a fresh tracker instance and installs it into the ambient/thread-local context; the tracker is retrievable from inside the `with` block via that same context mechanism.
- [x] Invalid pricing passed to the tracker raises before the `with` block's body ever runs.
- [x] An exception raised inside the `with` block propagates normally — never swallowed.
- [x] **Documented current behavior, not necessarily desirable**: exiting the `with` block does **not** reset the ambient context back to `None` (or restore a prior value) — the just-used tracker instance remains retrievable afterward. A port should decide deliberately whether to keep or fix this.

## Enums: `ModelBackend`, `ServiceProvider`, `ServiceType`
*`tests/llmfy_core/test_model_backend.py`, `test_service_provider.py`, `test_service_type.py`*

- [x] `ModelBackend` has exactly 5 members, one per (vendor, API-variant) pair, each following a `VENDOR_APIVARIANT` naming convention matching its string value: `openai_chat`, `openai_responses`, `bedrock_converse`, `google_generate`, `anthropic_messages`.
- [x] `ServiceProvider` has exactly 4 members (one per vendor, no API-variant split): `openai`, `bedrock`, `google`, `anthropic`.
- [x] `ServiceType` has exactly 2 members: `llm`, `embedding`.
- [x] All three: string-enum semantics (compare equal to their plain string value), and an invalid value is rejected.

## Responses
*`tests/llmfy_core/responses/`*

- [x] `AIResponse`: all 3 fields (`content`, `thinking`, `tool_calls`) are optional/nullable; unknown extra fields rejected.
- [x] `GenerationResponse`: `result` is required; `messages` defaults to an empty list, and that default is **not** a shared mutable default across instances (a common footgun in several languages' default-value semantics — verify explicitly in a port). Unknown extra fields rejected.

## LLM provider formatters
*`tests/llmfy_core/llms/{openai/chat,openai/responses,anthropic/messages,bedrock/converse,google/generate}/test_*_formatter.py`*

These are the highest-value, highest-bug-risk logic in the whole package —
tested exhaustively for **all 5 backends**. This section is the closest
thing to a spec for "how does a `Message` become provider wire JSON."

**Cross-provider behaviors to specifically watch for** (each already covered per-provider by name below, called out together here because they're easy to get subtly wrong when porting one provider at a time without comparing against the others):
- [x] **Tool-result merge vs. new-message asymmetry**: Anthropic and Bedrock *merge* multiple tool results from the same assistant turn (same `request_call_id`) into a single tool-role message; OpenAI (both APIs) and Google *always* create a brand-new message per tool result, never merging.
- [x] **Required-tool-parameter asymmetry**: OpenAI marks every parameter required regardless of a Python default (because `strict=True` is hardcoded); Anthropic and Bedrock also mark every parameter unconditionally required (no `strict` concept at all); Google is the **only** one of the five that respects a parameter's default value when deciding required-ness.
- [x] **Role mapping for tool-result messages**: every non-OpenAI-Chat provider maps the internal `tool` role to that provider's own conversational role for tool results (`"user"` for Anthropic/Bedrock/Google) rather than a distinct `tool` role — OpenAI Chat is the only one with a genuine native `tool` role.
- [x] Every provider (except Bedrock/Google, which support VIDEO) raises a clear, provider-specific error for unsupported content — e.g. OpenAI Chat/Responses reject `VIDEO`, Responses also rejects `DOCUMENT`, Anthropic rejects `VIDEO`.
- [x] A `Union[X, None]` (i.e. `X | None`) Python type hint on a tool parameter unwraps to `X`'s mapped JSON type in every formatter, not a generic fallback.

**OpenAI Chat** (`test_openai_chat_formatter.py`)
- [x] Text/image/document content blocks format correctly; `DOCUMENT` requires a `filename` (raises if absent); `VIDEO` always raises.
- [x] Tool calls format as OpenAI's `{"id","type":"function","function":{"name","arguments"}}` shape (arguments JSON-serialized).
- [x] A tool-result message uses **only the first** item of `tool_results` as the message content (documented single-result assumption for this API).
- [x] Tool-function schema: every parameter is required (via `strict=True`); a default value is appended to the parameter's description text; the `self` parameter is skipped for bound methods; an unmapped Python type falls back to `"string"`.
- [x] `format_tool_message` always appends a brand-new message (see cross-provider note above).

**OpenAI Responses** (`test_openai_responses_formatter.py`)
- [x] A single `Message` with multiple tool calls must expand into **multiple** flat top-level API items — represented internally as a private `{"__items__": [...]}` wrapper that the model layer unwraps (this API's `input` array is flat, unlike Chat Completions' single nested message).
- [x] Prior assistant-turn text replays as `"output_text"`; every other role's text is `"input_text"`.
- [x] `DOCUMENT` content is **not supported at all** on this API variant (raises), unlike Chat Completions (which supports it).
- [x] Tool-function schema omits the `"type": "function"` wrapper key (the model layer adds it) but otherwise mirrors Chat Completions' required-params-always logic.

**Anthropic Messages** (`test_anthropic_messages_formatter.py`)
- [x] Image content requires an explicit, supported `format` (`jpeg`/`png`/`gif`/`webp`); raw bytes are base64-encoded inline; the Bedrock-only `use_s3` flag is explicitly rejected as unsupported on this native API.
- [x] `DOCUMENT` requires a `filename`; `VIDEO` always raises (not supported at all on this API).
- [x] The internal `name` field is **never** emitted (the native Messages API has no per-message name field).
- [x] Tool-result blocks use the field name `tool_use_id` (not `id`) and merge multiple parallel results from the same turn into one message's `tool_results` list.
- [x] Tool-function schema description falls back to the function's name when no docstring description is available.

**Bedrock Converse** (`test_bedrock_converse_formatter.py`)
- [x] Richest content-type support of the five: text/image/document/video, each with both a `bytes`-source and an `s3Location`-source variant (the latter requires `bucket_owner`, raises if absent).
- [x] Video supports Bedrock's own 9-format list (`wmv`/`mpg`/`mpeg`/`three_gp`/`flv`/`mp4`/`mov`/`mkv`/`webm`).
- [x] The `name` field is dropped specifically for tool-role messages but kept for every other role.
- [x] Tool-result merging follows the same same-`request_call_id` rule as Anthropic (both support this "batch tool results into one message" behavior, documented in-repo as supporting two historically-different wire-format generations).

**Google AI Generate** (`test_googleai_generate_formatter.py`)
- [x] Role mapping is distinctive: `assistant`→`"model"` (not `"assistant"`); a placeholder `"system"` role is emitted for system messages (the model layer strips it out and passes it separately, same pattern as Anthropic/Bedrock but with a different placeholder string).
- [x] Image content branches on 3 input shapes: raw bytes (base64-encoded inline, hardcoded `image/jpeg` mime type), a `data:` URI string (mime type parsed **from the URI header itself**), or an `http`-prefixed URL string (passed as a file reference) — anything else raises.
- [x] Document content supports the same 3 shapes **except** it does *not* parse the mime type from a data-URI header the way image does — it hardcodes `application/pdf` regardless. This is a known inconsistency in the reference implementation, worth a deliberate decision (reproduce as-is for parity, or fix) rather than accidentally diverging.
- [x] Video content defaults its format to `mp4` when unset, validates against a Google-specific 9-format list, and — unlike image/document — has **no** data-URI support (only raw bytes or an http URL).
- [x] Tool-function schema is the only one of the five that respects a Python default value for required-ness (see cross-provider note above).
- [x] `format_tool_message` never merges (always a new message) despite `format_message`'s tool-result branch reading the *whole* `tool_results` list — meaning any merging behavior for Google would have to happen before this call, not inside it.

## LLM provider model classes
*`tests/llmfy_core/llms/{openai/chat,openai/responses,anthropic/messages,bedrock/converse,google/generate}/test_*_model.py`*

Depth policy: **OpenAI Chat is tested in full** (it's the reference
implementation for the shared streaming/error-wrapping/async patterns every
other backend follows); the other four get a smoke pass (construction
validation + one successful call + one mapped-error call) since their
formatting-specific logic is already covered above.

**Shared behavior across all 5 model classes** (each verified at least once):
- [x] Constructing a model without the required API credentials (API key, or for Bedrock all three of access key / secret key / region) raises a clean, actionable `LLMfyException` naming the missing credential — never a raw SDK error.
- [x] A provider SDK error raised by the mocked client call surfaces as the **correctly-mapped** `LLMfyException` subclass (e.g. a rate-limit error becomes `RateLimitException`) — this proves the model-to-error-handler wiring end-to-end, not just the handler in isolation.
- [x] A non-provider, generic exception (e.g. a plain network failure) is wrapped as a generic `LLMfyException` rather than left unwrapped or misclassified.
- [x] A response with a tool-call/function-call result parses into `AIResponse.tool_calls` with `content=None`; a plain-text response parses into `AIResponse.content` with `tool_calls=None`.

**OpenAI Chat only** (`test_openai_chat_model.py`) — full depth:
- [x] Both sync (`generate`) and async (`agenerate`) paths tested against their respective client objects.
- [x] Streaming: a chunk with an empty `choices` list (the trailing usage-only chunk) is silently skipped, never yielded as a response.
- [x] Streaming: **two parallel tool calls, with their argument-JSON deltas interleaved across chunks**, are accumulated **independently keyed by each tool call's index** — proving the accumulator can't accidentally merge one tool call's arguments into another's when the model streams them concurrently.
- [x] `max_tokens` is omitted entirely from the request payload when unset (not sent as an explicit null) — some models reject an explicit null outright.
- [x] Passing `tools` adds `tool_choice: "auto"` and wraps each tool definition in the `{"type": "function", "function": ...}` envelope.

**System-message handling** (Anthropic + Bedrock model smoke tests):
- [x] A `system`-role message present in the *formatted* message list gets hoisted out into that provider's dedicated top-level `system` parameter before the API call, and is **not** present in the `messages`/`contents` array sent to the SDK.

## Embeddings
*`tests/llmfy_core/embeddings/test_openai_embedding.py`* (OpenAI only — Bedrock/Google embeddings are not yet covered by this suite; a good next addition, not done in this pass)

- [x] Construction validation mirrors the chat models (missing API key / missing SDK → clean `LLMfyException`).
- [x] `encode()` returns the first (and only) embedding vector from the response; an empty response raises `ValueError`; any other underlying error propagates **unwrapped** (not re-wrapped as `LLMfyException` — a deliberate contrast with the chat models).
- [x] `encode_batch()`: a single string input is wrapped into a one-item list; requires numpy installed (clean error if missing); batches respect the configured `batch_size` (verified via captured per-call batch sizes); a mismatched response length (fewer embeddings returned than texts sent) raises `ValueError`.
- [x] **Response ordering is not guaranteed by the provider** — `encode_batch` must re-sort returned embeddings by each item's `index` field before returning, verified with a deliberately out-of-order fake response.
- [x] Rate-limit errors (detected via substring match on the error message) are retried with exponential backoff up to `max_retries`, then re-raised if still failing after the last attempt.

## `LLMfy` — the main chat/generation class
*`tests/llmfy_core/test_llmfy.py`* — tested against a fake `BaseAIModel` test double, never a real provider.

- [x] Constructor validates a `{{placeholder}}`-containing system message against the supplied `input_variables` at **construction time**: a placeholder with no `input_variables` at all raises; a placeholder missing from a non-empty `input_variables` list raises; a fully matching set succeeds.
- [x] `register_tool` requires every function to already carry the `@Tool` marker — a plain undecorated function (even a lambda) raises a clean error.
- [x] `invoke`/`chat` **record** any tool calls the model returns in conversation history but **never execute them** — only the `_with_tools` variants execute tools. Verified explicitly: a tool registered to flip a flag is proven *not* to have been called.
- [x] A generic exception raised by the underlying model is wrapped as `LLMfyException`; an `LLMfyException` raised by the model (including its `status_code`) is re-raised as-is, not double-wrapped.
- [x] The system-message template is rendered at **call time** using kwargs passed to `invoke`/`chat` (not only validated at construction time) — a call missing a required kwarg raises even if construction-time validation passed (construction only checks the variable *names* line up, not that every call site remembers to supply them).
- [x] `invoke_with_tools`/`chat_with_tools`: the tool-calling loop terminates as soon as the model responds with no more tool calls — verified for both a single round-trip and a **multi-round** scenario (two separate tool-call rounds before a final content answer), proving the loop actually advances rather than looping forever or exiting too early.
- [x] A tool's return value is **stringified** before being recorded as the tool result message content, regardless of its original type.
- [x] Calling an unregistered tool name during the tool-calling loop raises a clean error (same guarantee as the standalone `ToolRegistry`).
- [x] `chat`/`chat_with_tools` replaying a caller-supplied message history dispatches each message by its role (`user`/`assistant`/`tool`) to the matching history-builder method; a tool-role message carrying **multiple** `tool_results` only replays the **first** one (a documented truncation, not a crash).
- [x] Streaming (`invoke_stream`/`chat_stream`): content chunks are yielded as they arrive and their text is accumulated; a **final terminal chunk** is always yielded after the stream ends, carrying an empty `AIResponse` (`content=None`) paired with the full accumulated message history — this is the signal consumers use to retrieve final history, and it must be distinguishable from a genuine empty-string content chunk (which is a separate, real edge case also verified: an empty-string chunk from the model is indistinguishable from "no content this chunk" by design).
- [x] `LLMfy` builds a fresh, unshared message history for every `invoke`/`chat` call (no `messages_temp`/`clear_messages_temp()` public attribute) — reusing one instance for a second call must not see the first call's history, so concurrent calls on the same instance no longer race on shared conversation state.
- [x] Every async method (`ainvoke`, `ainvoke_with_tools`, `achat`, `achat_with_tools`) mirrors its sync counterpart's behavior exactly, including the tool-execution-loop and exception-wrapping rules; `ainvoke_stream`/`achat_stream` are thread-offloaded wrappers around the sync streaming generator (not independently-implemented native async streams) and must be verified to produce identical output to their sync counterparts.

## Utilities: chunking & text preprocessing
*`tests/llmfy_utils/`*

**`chunk_text`** (`chunk/test_chunk.py`)
- [x] Splits on whitespace into words, then slides a window of `chunk_size` words with a step of `chunk_size − chunk_overlap`.
- [x] **Chunks whose joined character length is ≤100 are silently dropped** — even otherwise-valid short input can produce zero chunks. Chunk ids only increment for chunks actually kept, so a dropped chunk never "uses up" an id.
- [x] `chunk_overlap == chunk_size` (step of exactly 0) is an **error** condition; `chunk_overlap > chunk_size` (a negative step) instead silently produces an **empty result** with no error — two different failure modes for two similar-looking misconfigurations, both worth deciding deliberately in a port rather than picking one behavior arbitrarily.
- [x] Accepts either a plain string or a `(text, metadata)` tuple; non-dict metadata gets wrapped as `{"meta": metadata}` rather than rejected.
- [x] **Security/robustness**: `None` input fails cleanly rather than silently producing garbage; ~1MB of input processes without hanging; unicode text (including scripts with no whitespace, like CJK) doesn't crash, even though the whitespace-splitting approach is documented as imperfect for such scripts.

**`chunk_markdown_by_header`** (`chunk/test_chunk.py`)
- [x] Recognizes ATX-style headers (`# `, `## `, ... up to 6 `#`s) that have **exactly one space** after the hashes — `#NoSpace` is never matched. Setext-style headers (underline-style) are never supported.
- [x] Each chunk's content includes its header line and runs up to (but not including) the next header, or end of text.
- [x] An optional `header_level` filters which header depths start a new chunk — a deeper header nested under a shallower one becomes ordinary chunk content, not its own chunk.
- [x] Content appearing before the very first header in the document is dropped entirely (never a "preamble" chunk).
- [x] `header_level=0` is an invalid quantifier and fails cleanly at the regex-construction level; a **negative** `header_level` does *not* fail the same way — it silently compiles into a pattern that never matches anything, producing an empty result instead of an error. (This is the header-level equivalent of the `chunk_text` overlap-edge-case inconsistency above — same category of "two similar misconfigurations, two different failure shapes.")
- [x] No headers anywhere in the input produces an empty result, not an error.

**`clean_text_for_embedding`** (`text_preprocessing/test_text_preprocessing.py`)
- [x] Unicode-normalizes text (NFKC form) — e.g. full-width characters collapse to their standard-width equivalent.
- [x] Collapses every run of whitespace (including newlines and tabs) into a single regular space, then trims leading/trailing whitespace. **This destroys paragraph/line structure** — multi-paragraph input becomes one line — which is intentional for this function's purpose (preparing text for an embedding model) but must not be assumed to also be "safe" for anything that needs line structure preserved.
- [x] Non-string input (including `None`) fails cleanly rather than silently coercing or crashing further downstream.
- [x] ~1MB of input processes without hanging.

**`deprecated` decorator** (`test_deprecated.py`)
- [x] Works on both plain functions and classes; decorating a class wraps its `__init__` (warns on instantiation, not on class definition), and works even for a class with no explicit `__init__` of its own.
- [x] **Decoration itself never warns** — only actually calling the decorated function, or instantiating the decorated class, does.
- [x] The warning message assembles from whichever of `reason`/`version`/`alternative` were supplied, in a fixed order and format; a custom warning category is honored (defaults to `DeprecationWarning`).
- [x] Function metadata (`__name__`, `__doc__`) is preserved on the wrapped function.
- [x] Decorating a class returns the **same class object** (not a copy/subclass) — identity-preserving.

## FlowEngine: edges, nodes & graph compilation
*`tests/flow_engine/edge/`, `tests/flow_engine/node/`, `tests/flow_engine/graph/`, `tests/flow_engine/test_flow_engine_build_validate.py`*

**`Edge`** (`edge/test_edge.py`)
- [x] Constructor accepts a single target string or a list; a string target is normalized to a one-item list internally, a list target is kept as-is (both end up as the same `list[str]` shape on `.targets`).
- [x] `condition`/`target_map` both default to `None` and are stored independently of `targets` (setting one doesn't affect the other).

**`Node`** (`node/test_node.py`)
- [x] Minimal construction only requires `name`/`node_type`; `sources`/`targets` default to independent empty lists per instance (not a shared mutable default across instances — a common dataclass footgun). `retry`/`timeout` are stored as given.
- [x] `START` (`"__start__"`) and `END` (`"__end__"`) are distinct reserved string constants; `NodeType` has exactly 4 members (`START`, `END`, `FUNCTION`, `CONDITIONAL`).

**Graph compilation** (`graph/test_graph_builder.py`) — `build_graph()` turns `nodes`/`edges` into a `CompiledGraph` once at `build()` time, replacing a linear scan of `edges` on every transition:
- [x] `targets_of(node)` returns a node's static (non-conditional) outgoing targets; empty for a node with no outgoing edge or a conditional one (conditional targets are resolved at runtime, not statically).
- [x] Two separate `add_edge(src, a)` / `add_edge(src, b)` calls from the same source compile into the **same** fan-out target list as a single `add_edge(src, [a, b])` call; a duplicate target across such calls is deduplicated, not repeated.
- [x] `is_join(node)` is true only once a node has **more than one** non-conditional predecessor; a single predecessor, or a node reached only via a conditional edge, or `START` itself, never count as join predecessors — `START` is deliberately excluded so a node that's both the graph's entry point and a loop-back target doesn't wait forever for a second "arrival" that can never come, and a conditional edge's targets are excluded so a join never waits on a branch that (by definition, only one conditional target runs per pass) will never show up.
- [x] `is_conditional(node)` is false for a node whose only outgoing edge is a regular (non-conditional) one.

**Structural validation** (`graph/test_validation.py`) — `validate_workflow()`, run once at `build()`:
- [x] A valid linear workflow, and one where `END` is only reachable via a conditional edge, both pass.
- [x] Missing an edge out of `START`, missing any path to `END`, an edge referencing an undefined target node, and an edge sourced from an undefined node all raise.
- [x] A node that has **both** a regular edge and a conditional edge as its source raises (ambiguous: which one fires?).
- [x] A join node (multiple predecessors) whose predecessors are **all** regular edges passes; if even one of those predecessors reaches it via a conditional edge, it raises (a conditional predecessor might not actually run, so the join could wait forever).
- [x] A node with no path from `START` at all only **warns**, it does not raise — a workflow can contain a deliberately-unreachable node (e.g. dead code during iteration) without failing `build()`.

**`FlowEngine` builder methods** (`test_flow_engine_build_validate.py`)
- [x] `_extract_state_annotations`: a plain (non-`Annotated`) field has no reducer; an `Annotated[Type, reducer_fn]` field extracts both the reducer and the inner type; a reducer that isn't callable, or doesn't take exactly 2 parameters, raises at `FlowEngine(...)` construction — before any node runs.
- [x] `add_node`/`add_edge`/`add_conditional_edges` reject the reserved names `START`/`END` as a node name, reject `START` as any edge's target, reject `END` as any edge's source, and reject an edge that targets its own source (including inside a fan-out list) — all at call time, not deferred to `build()`.
- [x] `add_edge`/`add_conditional_edges` correctly update each node's `sources`/`targets` bookkeeping as edges are added, including for a fan-out list target.
- [x] `add_conditional_edges` marks its source node `CONDITIONAL` type automatically; the dict form (`{label: target}`) builds an edge whose `targets` is the dict's values and whose `target_map` is the dict itself; a dict value of `START` is rejected the same as the list form.
- [x] `build()` returns `self` and sets `is_built = True`; calling `invoke()`/`stream()`/`details()` before `build()` raises `GraphValidationException`; `build()` itself surfaces the same structural errors as direct `validate_workflow()` calls (missing `START`/`END` path, undefined node reference).
- [x] `details()` (after `build()`) lists every function/conditional node and every regular/conditional edge in a plain-text summary; `visualize()` returns a `mermaid.ink` URL.

## FlowEngine: execution loop
*`tests/flow_engine/execution/test_engine_loop.py`* — the core rewrite; these are the highest-value tests in the whole suite, proving the concurrency bug is fixed and every new capability (fan-out, retry, timeout, hooks, checkpoint metadata) behaves as documented.

**Linear execution & reducers**
- [x] A node's returned dict updates flow through each field's reducer (or replace, if unannotated) into `ctx.state`; `ctx.step` increments once per node executed (not per edge or per event).

**Static fan-out / fan-in**
- [x] A diamond graph (`fetch` → `[branch_a, branch_b]` → `combine`) runs `combine` **exactly once**, only after both branches complete, regardless of which branch finishes first (verified with both orderings of a `sleep()`-delayed branch) — reducers (e.g. a set-union reducer) let both branches' updates land without clobbering each other.
- [x] Branches of a fan-out execute **concurrently, not sequentially** — verified by interleaved start/end ordering with different `sleep()` durations per branch (this is the specific concurrency bug the rewrite fixes).

**Concurrency safety across calls**
- [x] Two `invoke()`-equivalent runs on the **same** `EngineLoop` instance, driven concurrently via `asyncio.gather`, never corrupt each other's state — each `ExecutionContext` is independent per call.

**Step limit**
- [x] A conditional self-loop with no exit condition trips `StepLimitExceededException` once `ctx.step` reaches the configured `max_steps` (the exception carries `max_steps` and `session_id`); a loop that terminates before the limit completes normally.

**Retry** (`RetryPolicy`)
- [x] A node that fails transiently succeeds once attempts reach a matching result, retried up to `max_attempts`; exhausting all attempts raises `NodeExecutionException` carrying `node_name`, the final `attempt` count, and the original exception as `__cause__`. The default policy (no `retry=` passed) means exactly one attempt, no retry.

**Timeout**
- [x] A node exceeding its configured per-attempt `timeout` raises `NodeTimeoutException` carrying `node_name` and `timeout_seconds`.

**Hooks**
- [x] `on_node_start`/`on_node_end` each fire exactly once per node, in order, with the expected `(name, state)`/`(name, state, updates)` arguments; `on_error` fires with `(name, exception_type)` and the exception still propagates afterward (the hook observes, never suppresses).

**Streaming nodes**
- [x] A `stream=True` node's `STREAM`-type yields surface as `node_stream` events in order, and its single terminal `RESULT`-type yield becomes exactly one `node_result` event whose `state` merges into `ctx.state`; a stream node yielding anything other than a `NodeStreamResponse` raises `GraphValidationException`.

**Checkpointer integration**
- [x] `InMemoryCheckpointer` (`requires_serialization = False`) accepts an unregistered custom object in state and round-trips it via deep-copy with no type registry needed. A checkpointer with `requires_serialization = True` (standing in for Redis/SQL) raises `CheckpointDeserializationException` for the same unregistered type, and saves it as a safe, codec-tagged dict (`{"__type__": ..., "data": ...}`) once the type is registered via a `TypeRegistry`.

**Checkpoint metadata: `prev_node`**
- [x] A run's first node records `prev_node = START`; a linear chain's second node records its actual predecessor's name; every branch of a *static* fan-out records the fan-out source as `prev_node`; a *dynamic* (`Send`) fan-out's single aggregate checkpoint records the node that dispatched the `Send`s (not any individual branch — branches don't get their own checkpoints at all).

**Checkpoint metadata: `attempt` / `updated_fields` / `dispatch_id`**
- [x] `attempt` is `1` on a first-try success, and reflects the actually-winning retry count on both a plain node and a streaming node (the streaming path threads the winning attempt back via a separate mutable holder, since an async generator can't itself return a value — a distinct code path worth its own test).
- [x] `dispatch_id` is `None` for every regular (non-`Send`) checkpoint.
- [x] `updated_fields` exactly matches the node's own returned keys (`[]` when a node returns nothing).
- [x] A dynamic fan-out's single aggregate checkpoint has `attempt = None` (no single attempt count applies across N independently-retried branches committed together) but a non-`None` `dispatch_id`, and `updated_fields` is the union of every branch's returned keys.

**Dynamic fan-out (`Send`)**
- [x] N `Send`s to the same node run once per `Send` and reduce (join downstream) exactly once, regardless of N; an empty `list[Send]` from the router produces no branch execution and no further graph advance at all; a single bare `Send` (not wrapped in a list) is accepted the same as a one-item list.
- [x] A `Send` targeting a node not declared in that conditional edge's `targets` raises `GraphValidationException` naming the undeclared target; `Send`s in one routing call targeting two *different* declared nodes raises ("must target the same node"); a list mixing `Send` and non-`Send` items raises ("non-Send item").
- [x] `Send.state` fully **replaces** the branch's input — the branch function never sees the router's parent state, only what was passed to `Send(...)`; the branch's own **returned** updates still merge into shared state normally via the field's reducer.
- [x] `on_node_start`/`on_node_end` hooks and stream events for a branch reflect only that branch's own (replaced) state, never the parent's shared state — even though a downstream reader (e.g. a later `on_node_end`) sees the *dispatch's* state as it looked **before** commit (see next bullet).
- [x] Sizing `max_steps`: a dispatch of N `Send`s counts the router node plus all N branches toward the step count (e.g. router + 10 branches against `max_steps=5` trips `StepLimitExceededException`).
- [x] `retry`/`timeout` configured on the target node apply **independently per branch** (a flaky branch retries on its own; one slow branch timing out doesn't wait for/affect siblings).
- [x] `on_node_start`/`on_node_end` fire once per branch (not once per dispatch); `on_branch_start`/`on_branch_end` fire *in addition*, also once per branch, carrying `(name, branch_index, branch_total, ...)` — and are **not** fired at all for a plain node or a static fan-out branch (those already have distinct node names, so there's nothing to correlate).
- [x] **Atomic commit**: if any one branch raises, `NodeExecutionException` propagates and `ctx.state` is left **completely untouched** by the dispatch — including updates from sibling branches that finished successfully before the failure (verified with a fast, successfully-completing branch racing a slow, failing one). Exactly **one** checkpoint is saved for a successful N-branch dispatch (never N, never N+1 for N branches plus a downstream join node correctly getting its own); a failed dispatch saves **no** checkpoint for the dispatching node at all.
- [x] `on_node_end` for a branch sees state **as of just before** the dispatch's atomic commit — a field the branch itself just set is not yet visible in the `state` argument that hook call receives, even though it *is* visible on `ctx.state` right after the whole dispatch finishes.
- [x] `branch_index`/`branch_total`/`dispatch_id` populate correctly across a dispatch's `node_result` stream events (indices cover `0..N-1` exactly once, every event shares one `branch_total` and one non-`None` `dispatch_id`) and differ between two separate dispatches (even to the same node, even across different sessions); all three are `None` for every event outside a dynamic fan-out (plain nodes, static fan-out).

## FlowEngine: `RetryPolicy` & `FlowEngineHooks` (unit-level)
*`tests/flow_engine/execution/test_policy.py`, `tests/flow_engine/execution/test_hooks.py`*

**`RetryPolicy`**
- [x] Defaults: `max_attempts=1` (no retry), `retry_on=(Exception,)`, `backoff_seconds=0.0`.
- [x] `should_retry(exc)` matches configured exception types (and their subclasses) via `isinstance`, and rejects unmatched types.
- [x] `delay_for_attempt(n)` grows exponentially by `backoff_multiplier` per attempt past the first, honors a custom multiplier, and is always `0.0` regardless of attempt number when `backoff_seconds` is `0`.

**`FlowEngineHooks`**
- [x] All 5 callback fields (`on_node_start`, `on_node_end`, `on_error`, `on_branch_start`, `on_branch_end`) default to `None`.
- [x] Both sync and async callables are accepted and stored as given (dispatch-time awaiting is the engine loop's job, not validated here).
- [x] `on_error` receives exactly `(node_name, exception)`.

## FlowEngine: `Send` (dynamic fan-out primitive)
*`tests/flow_engine/execution/test_send.py`*

- [x] Constructor stores `node`/`state` as given; two `Send`s are equal iff both `node` and `state` match, unequal if either differs.
- [x] `node`/`state` cannot be reassigned after construction (immutable) — a deliberate guarantee that a `Send` dispatched into the engine can't be mutated out from under it mid-flight.

## FlowEngine: public `invoke()`/`stream()` API
*`tests/flow_engine/test_flow_engine_execution.py`, `tests/flow_engine/test_flow_engine_stream.py`*

- [x] `invoke()` returns the final state dict; `apply_state` seeds the initial state before the first node runs.
- [x] A conditional edge routes to its declared target both in list form (return value is the target name) and dict form (return value is a key into `target_map`); an unrecognized dict-form return value raises. A loop-back edge to the entry node keeps looping until the condition routes to `END`.
- [x] Static fan-out and `Send`-based dynamic fan-out both work end-to-end through the public `add_edge`/`add_conditional_edges` API (not just the lower-level `EngineLoop` used by the execution-loop tests above); `stream()` events for a `Send` dispatch carry the branch correlation ids described above.
- [x] The instance's `max_steps` (constructor default) applies when a call doesn't override it; a per-call `max_steps` argument overrides the instance default for that one call only.
- [x] `add_node(..., retry=...)` retries a transiently-failing node through the full public `invoke()` path (not just the engine loop directly); `FlowEngineHooks` fire during a real `invoke()` call.
- [x] Two different `session_id`s driven through the same built `FlowEngine` do not corrupt each other's state (the public-API-level equivalent of the concurrency-safety test above).
- [x] `stream()` yields a `start` event, then one `result` event per node for a non-streaming node; a `stream=True` node yields its `STREAM` chunks as `stream`-type events followed by one `result` event. A loop-back edge to the entry node keeps looping under `stream()` the same as under `invoke()`.

## FlowEngine: tool-calling helpers
*`tests/flow_engine/helper/`*

**`tool_trim_messages`** (`test_messages_trimmer.py`) — see the function's own docstring in `helper/messages_trimmer/messages_trimmer.py` for the full forward-scan mechanism; tests pin these outcomes:
- [x] A single-message history is returned as-is (too short to trim).
- [x] With no active tool cycle (last message isn't a tool result and no tool call is unresolved) — including right after a fully-resolved tool cycle followed by plain conversation turns — history is aggressively trimmed to just the last message.
- [x] A **pending** tool call (no result yet) protects every message from that tool-calling `ASSISTANT` message onward; the trimmable prefix before it anchors to the last **non-`TOOL`** message so the result never starts with an orphaned tool result — including when that means returning only the protected suffix (nothing usable before it) or skipping past several consecutive prior tool results to find a real anchor.
- [x] The **last** message being a tool result also protects context from its triggering `ASSISTANT` message onward, even if every tool call is already "resolved" (a just-completed round still needs its context preserved for the orchestrator to process next).
- [x] **Parallel tool calls** in one `ASSISTANT` message (several `tool_calls` in a single turn) are all preserved together — the protection boundary is the assistant message, not the individual tool call.

**`tools_node` / `tools_stream_node`** (`test_tools_node.py`)
- [x] `tools_node` executes every pending tool call from the last message via the registry and returns one `Message(role=TOOL)` per call, matching each call's `tool_call_id`; with no pending tool calls it returns an empty list.
- [x] `tools_node` does not mutate the caller's input `messages` list (works off a defensive deep copy).
- [x] `tools_stream_node` yields `EXECUTING` (name + arguments) before each tool call runs, then `RESULT` (carrying the `Message`) after — and yields nothing at all when there are no pending tool calls.

## FlowEngine: streaming response types
*`tests/flow_engine/stream/`*

**`FlowEngineStreamType`** (`test_flow_engine_stream_response.py`)
- [x] A `str`-enum, same as the other stream-type enums in the package; members compare equal to their plain string value; exact member set/values pinned.
- [x] `FlowEngineStreamResponse`'s fields (`type`, `node`, `content`, `state`, `error`, `branch_index`, `branch_total`, `dispatch_id`) all default to `None`; each is settable at construction.

## FlowEngine: visualizer
*`tests/flow_engine/visualizer/test_visualizer.py`*

- [x] The generated Mermaid diagram includes every node's name.
- [x] A regular edge renders as a solid arrow; a conditional edge renders as a dashed arrow labeled with the condition's possible return values; a dict-form (`target_map`) conditional edge labels each arrow with its dict key rather than a generic label.
- [x] A fan-out edge (list target) produces one arrow per target, not a single merged arrow.
- [x] `generate_diagram_url` produces a valid `mermaid.ink` URL with the diagram base64-encoded into it.

## FlowEngine: checkpoint storage (`InMemoryCheckpointer`, `SQLCheckpointer`, `RedisCheckpointer`)
*`tests/flow_engine/checkpointer/`, `tests/flow_engine/test_flow_engine_checkpoint_resume.py`*

**Custom-type registry** (`checkpointer/serde.py`, `test_serde.py`) — what `StateCodec`'s tagging (below) is built on:
- [x] `TypeRegistry` registers and resolves both a Pydantic `BaseModel` and a `@dataclass` by name (also accepts an initial `types=[...]` list at construction); resolving an unregistered name returns `None` rather than raising; registering a plain class (neither a `BaseModel` nor a `@dataclass`) raises.
- [x] `serialize_state`: JSON-native values (str/int/float/bool/None/list/dict) pass through untouched; a registered type anywhere in state — top-level, nested inside a dataclass, or inside a list — is tagged as `{"__type__": ..., "data": ...}`; an **unregistered** custom type, or any other unsupported type, raises at save time (never silently drops data).
- [x] `deserialize_state` round-trips a registered Pydantic model, dataclass, a dataclass nested inside another, and a list of a registered type, back to real objects; an unregistered type tag raises at load time; a tampered/unknown `__type__` tag raises **without ever attempting to import or instantiate** whatever it names (no arbitrary-code-execution surface from a corrupted or maliciously-crafted checkpoint); a plain dict with no type tag at all passes through unchanged.

**Wire-format codec** (`checkpointer/codec.py`, `test_codec.py`)
- [x] `StateCodec` round-trips a JSON-safe state dict through every combination of zlib compression and Fernet encryption (both off, either alone, both together) — order is compress-then-encrypt on encode, reversed on decode, since ciphertext doesn't compress.
- [x] Encrypted output does not contain any recognizable plaintext substring from the original state.
- [x] A serialized state exceeding the configured `max_state_bytes` raises `CheckpointPayloadTooLargeException` **before** anything is written; `max_state_bytes=None` disables the check entirely.
- [x] Constructing a codec with `encryption_key` set but the `cryptography` package unavailable raises a clear, actionable `LLMfyException` (install instructions), not a bare `ImportError`; constructing one with no `encryption_key` never even attempts the import.

**Retention (count-cap + TTL)** — all three backends
- [x] `max_checkpoints_per_session`: only the newest N checkpoints survive a `save()`; older ones become unreachable by `load(checkpoint_id=...)`. Unset (`None`, the default) preserves the pre-existing unbounded behavior. Enforcement is scoped per `session_id` — pruning one session never touches another's checkpoints.
- [x] `ttl_seconds`: checkpoints older than the configured age are dropped on the next `save()` for that session; combinable with `max_checkpoints_per_session` at the same time. `InMemoryCheckpointer` checks age against wall-clock time (no native expiry); `SQLCheckpointer` runs an indexed `DELETE ... WHERE timestamp < cutoff` scoped to the session (with `synchronize_session=False`, since these deletes never need to reconcile with in-memory ORM objects — evaluating them locally instead risks comparing a naive datetime, as reloaded by SQLite after a commit, against Python's timezone-aware cutoff); `RedisCheckpointer`'s pre-existing `ttl` remains a **sliding, whole-session** expiry (refreshed via `EXPIRE` on every save) — a distinct mechanism from per-checkpoint count/age retention, not a replacement for it.

**SQL storage** (`sql_checkpointer.py`, `test_sql_checkpointer.py`)
- [x] `state` is stored as a codec-encoded binary column (`LargeBinary`, `LONGBLOB` on MySQL to avoid its 64KB plain-`BLOB` cap) rather than JSON text — encryption round-trips end-to-end through the real column type, and decrypting with the wrong Fernet key fails loudly (`InvalidToken`) rather than silently returning garbage.
- [x] `echo=True` (SQL statement logging) is documented as unsafe for production once `encryption_key` isn't set, since it logs bound parameters as-is.

**Redis storage** (`redis_checkpointer.py`, `test_redis_checkpointer.py`)
- [x] Checkpoint metadata (id, session, timestamp, node, step) stays plain JSON; only the `state` field is codec-encoded then base64-wrapped so it can still live inside the same JSON payload — `list()` (which calls `load()` per entry) transparently benefits from decoding, compression, and decryption without its own logic.
- [x] Count-cap retention deletes both the overflowing checkpoints' data keys and their sorted-set membership together, so neither survives independently of the other.

**`session_id` validation** (`flow_engine.py`, `test_flow_engine_checkpoint_resume.py`)
- [x] A caller-supplied `session_id` must match `^[A-Za-z0-9_.:-]{1,255}$` or `FlowEngine.invoke()`/`stream()` raises `InvalidSessionIdException` before ever reaching a checkpointer — this is the single enforcement point shared by all three backends, since a malformed id is a write-path risk (unexpected Redis key-namespace segments, oversized SQL primary keys), not a read-path one.
- [x] An **empty string** `session_id` is falsy and is replaced by an auto-generated `uuid4()` before the regex check ever runs — same pre-existing treatment as `None`, not a new rejection case.
- [x] Engine-generated `checkpoint_id`s (always `uuid.uuid4()`) never pass through this check — only a caller-supplied `session_id` does.
