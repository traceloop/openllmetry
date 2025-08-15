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
All packages use Poetry as the package manager. Always execute commands through Poetry:
```bash
poetry run <command>
```

## Testing with VCR Cassettes
Tests utilize VCR cassettes for API calls.

### Commands
```bash
# Run tests normally (uses existing cassettes)
poetry run pytest tests/

# Re-record all cassettes (requires API keys)
poetry run pytest tests/ --record-mode=all

# Record only new test episodes
poetry run pytest tests/ --record-mode=new_episodes

# Record cassettes once (if they don't exist)
poetry run pytest tests/ --record-mode=once

# Run tests without recording (fails if cassettes missing)
poetry run pytest tests/ --record-mode=none

# Run specific test files
poetry run pytest tests/test_agents.py --record-mode=once
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
Flake8 is used for code linting.