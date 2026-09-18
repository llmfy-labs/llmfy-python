# Porting LLMfy to other language SDKs

This document is a recommendation, not a commitment or a roadmap with dates —
it exists because a second (and third, ...) language implementation of
LLMfy is a realistic near-term possibility, and several things about how
this Python implementation is currently built would make that port
error-prone if not addressed first. It's written from the perspective of
"what would make it hard for someone porting this to Rust/TypeScript/Go to
get the *exact same behavior*, not just the same shape."

## Why this matters now

Right now, LLMfy-the-concept exists only as LLMfy-the-Python-package: every
behavioral decision — how a tool result gets merged into conversation
history for Anthropic vs. how it doesn't for OpenAI, which provider makes a
schema parameter required-by-default vs. respects Python's own default
value, how cache discounts are computed for Google's tiered pricing — lives
exclusively as Python source code and, in places, only as an inline code
comment. A second SDK author has no source of truth to port from except
"read the Python and hope you didn't miss a branch." The goal of everything
below is to turn as much of that tribal knowledge as possible into an
artifact that isn't Python source, so a port becomes a translation exercise
against a spec, not a re-derivation of behavior via source-reading and
guesswork.

## 1. A language-agnostic behavior spec (fixtures, not just docs)

The formatter tests added in this pass (`tests/llmfy_core/llms/*/test_*_formatter.py`)
already enumerate, per provider, exactly how a `Message` (or a tool
definition) turns into that provider's wire format — including every content
type, every validation error, and the tool-result merge-vs-new-message
asymmetry. The recommendation is to promote a subset of these into checked-in
JSON fixtures: input `Message`/tool metadata → expected output dict, one file
per provider per scenario (text, image, tool call, parallel tool calls,
error cases). A port's own test suite loads the *same* JSON fixtures and
asserts its formatter produces byte-identical output. This turns "does the
Rust formatter behave like the Python one" into a mechanical diff instead of
a side-by-side reading of two codebases in two languages.

## 2. Externalize pricing tables and format-validation lists out of Python source

`OPENAI_PRICING`, `ANTHROPIC_PRICING`, `BEDROCK_PRICING`, `GOOGLEAI_PRICING`
(under `llmfy_core/llms/*/`.\*`_pricing_list.py`) are plain Python dicts, and
the provider-specific validation lists (allowed Bedrock image formats
`["gif", "jpeg", "png", "webp"]`, allowed video formats, etc., scattered
across each formatter) are Python literals embedded in `if`/`raise` branches.
Every other-language SDK currently has no way to consume these except
hand-transcribing them, and hand-transcribed data drifts — a price change or
a newly-supported format lands in the Python package and nowhere else.
Moving these into plain JSON/YAML data files that all SDKs (Python included)
load at runtime means a price update is a one-line data change reflected
everywhere, not an N-times reimplementation.

## 3. Document the exception taxonomy as a versioned, stable contract

`LLMfyException` and its flat one-level hierarchy (`RateLimitException`,
`TimeoutException`, ...), plus `exception_mapper.py`'s four provider
error-code → exception-class tables, are exactly the kind of thing every
SDK needs to reproduce identically so that application code written against
"catch `RateLimitException`" behaves the same regardless of which language's
SDK it's running on. Today this taxonomy is discoverable only by reading
`llmfy/exception/llmfy_exception.py` and `exception_mapper.py`. Worth writing
up as its own short spec (this repo's `technical-docs.md` already has the
raw material — it just needs to be framed as a cross-language contract,
including exactly which provider error codes map to which exception, and
what `status_code`/`provider`/`timeout_type` mean) so a port's error handling
can be verified against that spec rather than against a second reading of
Python's `exception_handler.py`.

## 4. Write down the currently-undocumented behavioral quirks

This test-writing pass surfaced several real, working-as-intended behaviors
that exist only as source-level detail — exactly the kind of thing a second
implementation gets subtly wrong because there's nothing prompting the
author to even think about it:

- Anthropic and Bedrock **merge** multiple tool results from the same
  assistant turn into a single tool-role message (keyed by
  `request_call_id`); OpenAI and Google always create a **new** message per
  tool result.
- Required-vs-optional tool parameters differ per provider: OpenAI marks
  every parameter required (because `strict=True` is hardcoded), Anthropic
  and Bedrock also mark every parameter required unconditionally, but Google
  is the only one that actually respects a Python default value.
- `LLMfyUsage(openai_pricing={})` — an explicitly empty dict — skips
  structure validation *and* silently falls back to the bundled default
  pricing table, rather than using an empty table. `None` and `{}` are not
  equivalent to a caller in the way they might expect.
- `MessageBufferBuilder.clear()` keeps only the most-recently-inserted system message
  if multiple were ever added (since `add_system_message` always inserts at
  index 0) — not "the first one," not "all of them."
- Google's cache-discount math computes the *full* input price first and
  then subtracts a savings delta, while OpenAI/Bedrock/Anthropic compute a
  *non-cached* input price directly — different formulas that are
  mathematically equivalent for the same discount ratio, but a port that
  "simplifies" one to match the other's shape will get the wrong number the
  moment the discount ratio isn't the assumed default.

None of this is a defect — it's intentional, tested behavior now (see the
formatter and `LLMfyUsage` test files added in this pass) — but it needs to
live in prose somewhere a porting author will actually read, not just in
Python comments and test assertions.

## 5. A recommended compatibility/versioning statement across languages

The current auto-tagging (`CONTRIBUTING.md`'s commit-prefix-driven semver
bump) is a Python-package release mechanic, not a cross-language compatibility
contract. Once a second language SDK exists, "LLMfy v0.10.0" needs to mean
the same set of observable behaviors regardless of which language's package a
user installs — otherwise a user switching from the Python SDK to (say) a
future TypeScript SDK at "the same version" can get different formatting or
pricing behavior with no warning. Worth a short explicit statement (even a
single paragraph in `technical-docs.md`) defining what a version number
promises across languages, and which behaviors are load-bearing enough to
require a coordinated bump everywhere versus which can drift per-language
(e.g. purely SDK-ergonomic APIs).

## 6. This pytest suite is reference material — and now has a language-agnostic index

Beyond the fixture idea in (1), the full test suite added in this pass
(`tests/`) enumerates essentially every observable behavior this package
guarantees, module by module. Reading pytest source to extract that,
though, requires Python fluency (fixtures, `unittest.mock`, `monkeypatch`,
pytest idioms) that a porting author in another language may not have and
shouldn't need.

**`TEST_CATALOG.md`** (new, sibling to this file) is that index: a
plain-language, checkbox-per-behavior checklist covering every test in
`tests/`, organized to mirror the test suite's own structure, with each
section naming its source test file(s) for traceability. A porting author
copies it into the new SDK's repo and checks items off as their own test
suite proves each one — no box should stay unchecked silently; an
unreproduced behavior should be a deliberate, written-down decision, not an
oversight. It also surfaces, in one place, several intentional cross-provider
asymmetries (tool-result merge-vs-new-message, required-parameter rules,
Google's differently-shaped cache-discount formula) that are exactly the
kind of detail a second implementation gets subtly wrong without a prompt to
even think about them — see point 4 above for the fuller narrative version
of the same list.
