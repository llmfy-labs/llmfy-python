# Contributing

## Commit message format

Commit subjects follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short summary>

[optional body]
```

- `type` is one of: `feat`, `fix`, `refactor`, `chore`, `ci`, `docs`, `test`.
- For a breaking change, prefix the subject with `[breaking-changes]`, e.g. `[breaking-changes] refactor: consolidate thinking config`.
- Keep the summary in the imperative mood (e.g. "add", not "added"/"adds").
- The release workflow copies each commit's subject and body verbatim into the GitHub release changelog, so write both to be read standalone (see `.github/workflows/release.yml`).

## Version bump rules (automatic tagging)

Every push to `main` is scanned by `.github/workflows/auto-tag.yml`, which tags a new release automatically — no manual `git tag` needed. The bump is decided per commit subject, in this precedence order (highest across all new commits wins):

| Commit subject | Bump |
|---|---|
| `[breaking-changes] <type>: ...` **or** `<type>!: ...` / `<type>(scope)!: ...` | **MINOR** (temporarily — see note below) |
| `feat: ...` (no breaking marker) | **MINOR** |
| `fix: ...` (no breaking marker) | **PATCH** |
| `refactor:`, `chore:`, `ci:`, `docs:`, `test:` alone | no release (bundled into the next qualifying commit) |

**Temporary, pre-1.0 only**: a breaking marker is currently capped at MINOR instead of MAJOR (see the `TEMPORARY` comment in `.github/workflows/auto-tag.yml`) — the package is still churning through frequent breaking changes pre-1.0, and a MAJOR bump per breaking commit isn't wanted yet. Once the API stabilizes, this reverts to the standard Conventional Commits rule: a breaking marker always forces MAJOR regardless of type, e.g. `[breaking-changes] feat: ...` or `feat!: ...`, both equivalent.

## Build package

```sh
uv build
```

## Manual / backfill release

Tagging is automatic (see above). To manually cut or re-run a release, use the `workflow_dispatch` trigger on `release.yml` (GitHub Actions UI → "Release & Publish" → "Run workflow", tag input `vX.Y.Z`) instead of pushing a tag by hand.

## After deploy on local

After CI creates or moves the tag, your local tag ref may be stale. To sync:

```sh
git fetch --tags --force
```

The --force flag is needed because git fetch --tags alone won't update tags that already exist locally.

## Package Development on local

```sh
uv sync --group dev --group docs
```

or

```sh
uv sync --all-groups
```

## Check Lints

```sh
uvx ruff check --statistics . 2>&1 | tail -60
```

## Run Tests

Test dependencies live in the `test` dependency group (`pytest`, `pytest-asyncio`,
`pytest-mock`, `pytest-cov`, `pip-audit`):

```sh
uv sync --all-groups
```

Run the full suite:

```sh
uv run pytest
```

With a coverage report:

```sh
uv run pytest --cov=llmfy --cov-report=term-missing
```

Notes:

- `llmfy/flow_engine/` currently has no tests in `tests/` — it's excluded on
  purpose pending a planned refactor; don't add tests there until that lands.
- Provider SDKs (`openai`, `boto3`/`aioboto3`, `anthropic`, `google-genai`)
  are installed via the `dev`/`test` groups but never actually called over
  the network — tests construct real (lightweight) SDK client objects and
  mock only the network-call method (e.g. `client.chat.completions.create`).
  No real API keys or credentials are required to run the suite.
- Dependency security checks (not part of `pytest`, run separately — see
  "Testing policy" below, they're required for every change, not just
  dependency updates):

  ```sh
  uv run pip-audit   # scans the resolved environment for known CVEs
  uv lock --check    # confirms pyproject.toml and uv.lock are in sync
  ```

  If `pip-audit` flags something, see "Investigating and fixing a
  `pip-audit` finding" below before touching `pyproject.toml`.

## Testing policy — required for every change

Applies to every `feat`/`fix`/`refactor` change, whether written by a human
or an AI coding agent. None of this is optional cleanup — it's the
definition of "done". `.github/workflows/test.yml` runs the lint and test
steps below automatically on every PR and every push to `main` (Python 3.11
and 3.12), plus a non-blocking `pip-audit` job — but CI catching a mistake
is a safety net, not a substitute for running these yourself before
pushing. **Note**: CI running the checks doesn't by itself stop a PR from
being merged with a red check — that requires turning "lint" and "test" on
as required status checks under the repo's branch protection settings for
`main` (a GitHub setting, not something a workflow file controls), which
hasn't been configured as part of adding this workflow.

1. **Add or update tests** covering the change, in the `tests/` path that
   mirrors the `llmfy/` file(s) you touched. Check `TEST_CATALOG.md` first —
   it's a plain-language checklist of every behavior the suite already
   covers, so you can tell what's genuinely new versus already proven.
   Reuse `tests/conftest.py`'s shared fixtures (`FakeAIModel`,
   `provider_api_keys`) instead of hand-rolling new provider mocks.
2. **Run the full suite, not just the file(s) you touched**:
   ```sh
   uv run pytest
   ```
   This codebase shares formatters, enums, and the exception hierarchy
   across modules (see `CLAUDE.md`'s architecture notes) — a change scoped
   to one file can silently break another module's tests. A green run of
   only the tests near your change is not sufficient proof; a green run of
   the *whole* suite is what "didn't touch another module" actually means.
3. **Keep lint clean**:
   ```sh
   uvx ruff check --statistics .
   ```
4. **Run the dependency checks too — for every change, not only ones that
   touch `pyproject.toml`/`uv.lock`**:
   ```sh
   uv run pip-audit   # scans the resolved environment for known CVEs
   uv lock --check    # confirms pyproject.toml and uv.lock are in sync
   ```
   Both are fast. Report their actual output rather than skipping them
   because the change "wasn't about dependencies" — that's exactly the kind
   of drift these two catch early. If `pip-audit` flags something, see
   "Investigating and fixing a `pip-audit` finding" below.
5. `llmfy/flow_engine/` currently has no automated tests (pending a planned
   refactor) — if a change touches it, say so explicitly and verify it
   manually; don't skip verification just because the suite can't catch
   regressions there yet.
6. **Report honestly.** If a test fails, say so with the actual output. Never
   weaken an assertion, delete a test, or skip it just to force a passing
   result — fix the code, or explain why the test's expectation was wrong.

## Investigating and fixing a `pip-audit` finding

`uv run pip-audit` scans every package actually resolved in the environment
(direct **and** transitive) against the PyPA advisory database. `llmfy`
itself only ever declares one hard dependency (`pydantic`) — every package
that comes up in a finding is transitive, pulled in by an optional extra
(`openai`/`boto3`/`aioboto3`/`anthropic`/`google-genai`/`redis`/`SQLAlchemy`)
or by the `dev`/`docs`/`test` tooling groups. Also note: `uv.lock` is
**gitignored** in this repo — it's not shared/committed, so a finding here
reflects only your own local resolution at this moment, not something
frozen for every contributor or CI run.

**1. Find which extra/group actually pulls it in**, don't guess:

```sh
uv tree --invert --package <flagged-package> --all-groups
```

This flips the dependency tree and walks upward from the flagged package to
every path that reaches it, ending at `llmfy (group: ...)` /
`llmfy (extra: ...)`. Read every branch — the same package is often pulled
in by more than one path (e.g. `urllib3` via both `botocore` and
`requests`). Example, and how to read it:

```
cryptography v46.0.6
└── google-auth v2.56.2
    └── google-genai[requests] v2.16.0
        └── llmfy (extra: google-genai)
```

Root → leaf here means: `cryptography` is not something `llmfy` chose —
`google-genai` depends on `google-auth`, which depends on `cryptography`.
It only appears when the `google-genai` extra (or `dev`/`all`) is installed.

**2. Pick the fix based on where the trace ends**, don't reach for
`pyproject.toml` by default — adding the flagged package there as a direct
dependency is almost always wrong: `llmfy` doesn't use it directly, so
declaring it misrepresents the dependency contract and doesn't even
guarantee anything for downstream installs (see the third case below).

- **Trace ends in `dev`/`docs`/`test` group tooling** (e.g. `mkdocs-material`,
  `mkdocstrings`, `pip-audit`'s own `requests`/`cachecontrol`) — these never
  ship in the published wheel, so the fix is purely local: refresh the
  resolver's pick, no `pyproject.toml` edit needed.
  ```sh
  uv lock --upgrade-package <flagged-package>
  uv sync --all-groups
  uv run pip-audit   # confirm it's cleared
  ```
- **Trace ends in an optional extra a real user might install**
  (`openai`/`boto3`/`aioboto3`/`anthropic`/`google-genai`/etc.) — run the
  same `uv lock --upgrade-package`/`uv sync` refresh for your own local
  environment and CI, but understand its limit: since `uv.lock` isn't
  committed, doing this does **not** change what a downstream user's own
  resolver picks when they `pip install llmfy[extra]` — that's controlled
  entirely by the upstream package's (`google-genai`'s, `boto3`'s, ...) own
  pinned dependencies. There's nothing further to fix on `llmfy`'s side
  beyond confirming `llmfy`'s own extras have no accidental upper-bound pin
  artificially blocking a newer, patched release of that upstream package.
- **`uv lock --upgrade-package` can't reach a fixed version at all**
  (the resolver is stuck on an old release because a pinned parent
  package's own version range doesn't allow the newer transitive
  dependency) — only then edit `pyproject.toml`, and only the **parent**
  package's own minimum-version pin in its `dependency-groups`/
  `optional-dependencies` entry (e.g. bump `mkdocs-material` itself, not
  `pillow`). Re-run `uv lock --upgrade-package <parent>` afterward to
  confirm the transitive fix actually resolves.

**3. Always re-verify after any fix**, the same way as any other change
(see "Testing policy" above): `uv run pip-audit`, `uv run pytest`,
`uvx ruff check --statistics .`, `uv lock --check`.

**Does a local fix like this need to reach CI?** No, and it can't anyway —
`uv.lock` is gitignored, so a local `uv lock --upgrade-package` never gets
committed or shared. That's not a gap for this repo's actual release
pipeline: `.github/workflows/release.yml`'s `publish` job only runs
`uv build` then `uv publish` — it never runs `uv sync --all-groups`, so it
never installs the `dev`/`docs`/`test` groups or any optional extra in the
first place, and therefore never touches whatever transitive package a
`pip-audit` finding was about. There is currently no test/lint/audit job in
CI at all (`auto-tag.yml` and `release.yml` are the only workflows, and both
are purely about versioning/publishing) — if one is added later, it will run
its own fresh `uv sync` on every invocation and pick up whatever's newest on
PyPI at that time, with no dependency on what any contributor resolved
locally. The trade-off of not committing `uv.lock` is reproducibility across
environments/time, not security — that's a separate, deliberate decision to
revisit only if it becomes an actual problem.

## Mkdocs run on local

```sh
uv sync --group docs
```

```sh
# Serve on local
mkdocs serve

# Build docs
mkdocs build
```
