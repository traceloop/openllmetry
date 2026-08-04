# OpenLLMetry Repository Guide

## Repository Structure
This repository contains multiple PyPI-publishable packages organized and orchestrated using Nx workspace management.

### Nx Workspace Commands
```bash
# Run tests across all packages
nx run-many -t test

# Run linting across all packages
nx run-many -t lint

# Update lock files across all packages
nx run-many -t lock

# Run specific targets on specific packages
nx run <package-name>:test
nx run <package-name>:lint

# Show project graph
nx graph

# Show what's affected by changes
nx affected:test
nx affected:lint
```

## Package Management
All packages use uv as the package manager. Always execute commands through uv:
```bash
uv run <command>
```

## Testing with VCR Cassettes
Tests utilize VCR cassettes for API calls.

### Commands
The default way to run a package's tests is through Nx:
```bash
npx nx run <package-name>:test
```

Reach for `uv run pytest` directly, from inside the package directory, only when
Nx's `test` target is not the right tool for the job — e.g. picking a VCR
record mode, or running a single test file:
```bash
# Run tests normally (uses existing cassettes)
uv run pytest tests/

# Re-record all cassettes (requires API keys)
uv run pytest tests/ --record-mode=all

# Record only new test episodes
uv run pytest tests/ --record-mode=new_episodes

# Record cassettes once (if they don't exist)
uv run pytest tests/ --record-mode=once

# Run tests without recording (fails if cassettes missing)
uv run pytest tests/ --record-mode=none

# Run a single test file inside a package
uv run pytest tests/test_agents.py --record-mode=once
```

### Guidance
Re-record cassettes when API interactions change to ensure test accuracy.
Never commit secrets or PII. Scrub them using VCR filters (e.g., filter_headers, before_record) or your test framework's equivalent.
Store API keys only in environment variables/secure vaults; never in code or cassettes.
Typical record modes you may use: once, new_episodes, all, none (choose per test needs).
Creating new cassettes requires valid API keys (OpenAI, Anthropic, etc.); ask the user to provide them if needed.

## Debugging with Console Span Exporter
For debugging OpenTelemetry spans and hierarchy issues, use the console exporter:

```python
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from traceloop.sdk import Traceloop

Traceloop.init(
    app_name="debug-app",
    exporter=ConsoleSpanExporter(),
    # other config...
)
```

This outputs all spans to console in JSON format, showing trace IDs, span IDs, parent relationships, and attributes for debugging span hierarchy issues.

## Semantic Conventions
The semantic convention package follows the OpenTelemetry GenAI specification:
https://opentelemetry.io/docs/specs/semconv/gen-ai/

## Instrumentation Packages
Instrumentation packages should leverage the semantic conventions package. Their purpose is to instrument AI-related libraries and generate spans and tracing data compliant with OpenTelemetry semantic conventions.

## Code Quality
Ruff is used for code linting. Run it through Nx, the default path:
```bash
npx nx run <package-name>:lint
```
Configuration is in each package's pyproject.toml under `[tool.ruff]`.

## For agents working in this repo

This file is canonical and vendor-neutral. `CLAUDE.md` is a symlink to it — never
edit `CLAUDE.md` directly, and never put knowledge in `.claude/`, which holds only
pointers.

**Procedures** — how to carry out recurring tasks — live in `docs/ai/procedures/`:

| Task | Procedure |
|---|---|
| Fix a reported bug | [fix-instrumentation-bug.md](docs/ai/procedures/fix-instrumentation-bug.md) |
| Add a new instrumentation package | [add-instrumentation.md](docs/ai/procedures/add-instrumentation.md) |
| Record or re-record a VCR cassette | [record-cassette.md](docs/ai/procedures/record-cassette.md) |
| Work with the semconv contract | [semconv-conformance.md](docs/ai/procedures/semconv-conformance.md) |

**Knowledge** — what we have learned about these SDKs and this codebase — lives in
`docs/ai/knowledge/`, an [OKF](https://github.com/GoogleCloudPlatform/knowledge-catalog)
bundle. Start at [index.md](docs/ai/knowledge/index.md). Each concept carries
provenance and a `stale_after` date; ignore concepts whose date has passed.

The split is deliberate: **imperative content belongs here, empirical content belongs
in the knowledge bundle.** "Run `nx run <pkg>:test`" is imperative. "Anthropic returns
an empty content block on tool-only turns" is empirical — it rots when the SDK bumps,
so it needs the expiry and provenance the bundle provides.

There are deliberately no per-package `AGENTS.md` files. All 35 packages share the
same nx targets, so per-package instruction files would be boilerplate. Package-specific
findings go in the knowledge bundle instead. Create a per-package `AGENTS.md` only if a
package genuinely deviates from the standard workflow.