# AGENTS.md

Guidance for coding agents and humans working in the OpenLLMetry repo. Read it before changing code.

## Repo map

A monorepo of independently published PyPI packages, orchestrated by Nx (`nx.json`, `@nxlv/python` plugin) and managed per package with uv. Every package lives in `packages/<name>/` with its own `pyproject.toml`, `uv.lock`, `project.json`, and `tests/`. The Nx project name equals the directory name.

| Package(s) | Purpose | Source path |
|---|---|---|
| `opentelemetry-instrumentation-<lib>` (32 packages: `openai`, `anthropic`, `bedrock`, `langchain`, `llamaindex`, `mcp`, `chromadb`, ...) | Instrument one AI/vector-DB/framework library | `opentelemetry/instrumentation/<lib>/` (hyphens become underscores, e.g. `openai_agents`, `google_generativeai`) |
| `opentelemetry-semantic-conventions-ai` | Shared GenAI attribute/metric constants | `opentelemetry/semconv_ai/` |
| `traceloop-sdk` | `Traceloop.init()`, decorators, exporters | `traceloop/sdk/` |
| `sample-app` | Example apps; excluded from CI tests | `sample_app/` |

Versions are bumped repo-wide by commitizen (`.cz.toml`) in the release workflow. Do not edit `version` fields or `version.py` files by hand.

## Commands

Install Node deps once at the repo root with `npm ci`; `nx` is a local devDependency, so invoke it via `npx`. Every package defines these Nx targets in `project.json`:

| Target | Runs (in the package dir) |
|---|---|
| `install` | `uv sync --all-groups` |
| `lint` | `uv run ruff check .` |
| `test` | `uv run pytest tests/` |
| `lock` | `uv lock` |
| `build` / `build-release` | Build the wheel |

`traceloop-sdk` also has `type-check` (`uv run mypy traceloop/sdk`).

```bash
npx nx run opentelemetry-instrumentation-openai:install
npx nx run opentelemetry-instrumentation-openai:lint
npx nx run opentelemetry-instrumentation-openai:test
npx nx affected -t lint      # what CI runs: affected install, lint, type-check, build-release, test
npx nx run-many -t test      # all packages
npx nx graph
```

Inside a package directory, run tools through uv: `uv run pytest tests/test_x.py -k name`, `uv run ruff check .`.

**Ruff:** configured per package in `pyproject.toml` under `[tool.ruff]` (line length 120, `select = ["E", "F", "W"]`). Lint must pass before you push.

## Testing and VCR cassettes

Tests use pytest with `pytest-recording` (vcrpy). Tests that hit a provider API are marked `@pytest.mark.vcr` and replay recorded HTTP from cassettes.

- **Location:** `pytest-recording` stores cassettes next to the test module, at `<test dir>/cassettes/<test_module>/<test_name>.yaml`. Most packages use `tests/cassettes/...`; some nest further, e.g. `packages/opentelemetry-instrumentation-openai/tests/traces/cassettes/test_chat/`.
- **Record mode in CI:** CI (`.github/workflows/ci.yml`) runs `npx nx affected -t test` and passes no `--record-mode` flag, and no `pytest.ini`/`pyproject.toml` sets one. `pytest-recording` therefore uses its default, `none`: cassettes are replayed and any request missing from a cassette fails the test. A `vcr_config` may override this (e.g. `traceloop-sdk/tests/guardrails/conftest.py` sets `record_mode: "once"`).
- **Record modes** (pass `--record-mode=<mode>`; recording requires real API keys in env vars):

```bash
uv run pytest tests/                             # replay only (default: none)
uv run pytest tests/test_chat.py --record-mode=once          # record missing cassettes
uv run pytest tests/test_chat.py --record-mode=new_episodes  # append new interactions
uv run pytest tests/test_chat.py -k name --record-mode=all   # re-record from scratch
```

- **Never hand-write or hand-edit cassettes.** Record them against the real provider API. If you lack keys, ask the user for them; do not fabricate responses. Re-record when the API interaction a test exercises changes.
- **Scrub secrets and PII.** Each package's `tests/conftest.py` defines a module-scoped `vcr_config` fixture. Follow the existing pattern and extend it for any new credential-bearing header, query parameter, or body field:

```python
@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": [
        "authorization", "api-key", "openai-organization",
        "openai-project", "set-cookie", "x-request-id",
    ]}
```

Use `filter_query_parameters`, `before_record_request`, or `before_record_response` for secrets outside headers (see `packages/opentelemetry-instrumentation-langchain/tests/conftest.py`). Inspect every new or changed cassette for keys, tokens, emails, and account IDs before committing.

- Tests assert on spans via an `InMemorySpanExporter` fixture (`span_exporter`). Fixtures such as `instrument_with_content` / `instrument_with_no_content` set `TRACELOOP_TRACE_CONTENT` to cover both content-capture modes.

## Semantic conventions

- Follow the OpenTelemetry GenAI semantic conventions: https://opentelemetry.io/docs/specs/semconv/gen-ai/
- Use constants, never hard-coded attribute strings. The code uses two sources:
  - Upstream incubating conventions, the preferred source for `gen_ai.*` names: `from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes` (plus `opentelemetry.semconv._incubating.metrics` and `opentelemetry.semconv.attributes.error_attributes.ERROR_TYPE`).
  - This repo's `opentelemetry-semantic-conventions-ai` (`opentelemetry.semconv_ai`): `SpanAttributes`, `Meters`, `Events`, `EventAttributes`, `LLMRequestTypeValues`, `TraceloopSpanKindValues`, `GenAISystem`, `SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY`, for attributes not (yet) covered upstream.
- If a needed attribute exists in neither, add it to `opentelemetry-semantic-conventions-ai` rather than inlining a string.

## Instrumentation correctness

Instrumentation packages wrap library calls (via `wrapt`) to emit spans, metrics, and events. Every change must preserve:

- **No behavior change for the app.** Instrumentation-originated failures must not break application behavior, and underlying SDK exceptions and cancellation semantics must stay intact. Wrapped calls must not swallow provider errors (existing tests require them to propagate, e.g. `test_chat_exception` in `packages/opentelemetry-instrumentation-openai/tests/traces/test_chat.py` uses `pytest.raises`). Guard telemetry-only code with the package's `dont_throw` helper; never wrap the provider call itself in it. Record errors on the span (status `ERROR`, `exception` event, `error.type`) and re-raise.
- **Sync/async parity.** Fix and test both the sync and async paths (e.g. `test_chat_exception` / `test_chat_async_exception`).
- **Streaming lifecycle.** Spans end on exhaustion, early close, and error; wrappers preserve iterator/context-manager behavior. Stream wrappers (e.g. `ChatStream(ObjectProxy)` in the openai package) must keep `__iter__`/`__aiter__`, `__enter__`/`__aenter__`, `close()`, and any attributes the SDK object exposes.
- **Context propagation.** Create spans in the caller's current context so they nest under the parent; honor `_SUPPRESS_INSTRUMENTATION_KEY` and `SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY`; do not leak attached context across calls.
- **Content-capture privacy.** Prompt/completion capture respects the existing toggles. Each package gates content behind its own `should_send_prompts()` helper (usually in `utils.py`), which reads the `TRACELOOP_TRACE_CONTENT` env var (default `"true"`) and the `override_enable_content_tracing` context value. `Traceloop.init(trace_content=...)` sets that env var. Never record prompts, completions, tool arguments, or message content outside that check.

## Debugging spans

Print every span (trace/span IDs, parent links, attributes) as JSON to check hierarchy:

```python
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from traceloop.sdk import Traceloop

Traceloop.init(app_name="debug-app", exporter=ConsoleSpanExporter())
```

## Commits and pull requests

- Use Conventional Commits with the package as scope, matching history:
  `fix(openai): ...`, `fix(sdk): ...`, `feat(instrumentation): ...`, `fix(sdk,openai-agents): ...`, `docs: ...`. The PR title becomes the squash commit, so it must follow the same format.
- Keep PRs small and focused: one bug or feature, one package where possible.
- Fill the checklist in `.github/PULL_REQUEST_TEMPLATE.md` honestly: tests added, screenshots from an observability platform for instrumentation changes, conventional PR name, docs updated if applicable.
- Every fix needs a test that fails without it, covering sync and async (and streaming, if relevant).

## Don'ts

- Do not change dependencies (`pyproject.toml` dependency lists, `uv.lock`) in fix PRs. Propose dependency changes separately.
- Do not skip, delete, or weaken tests, and do not add `skip`/`xfail` to make things pass.
- Do not edit CI (`.github/workflows/`).
- Never commit secrets or PII: no API keys in code, cassettes, fixtures, or logs. Keep keys in environment variables only.
- Do not hand-write or hand-edit VCR cassettes.
- Do not hard-code semantic-convention attribute strings.
