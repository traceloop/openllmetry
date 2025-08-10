# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

This is the **traceloop-sdk** package within the OpenLLMetry monorepo - a comprehensive observability solution for LLM applications built on OpenTelemetry. The monorepo contains:

- **traceloop-sdk**: Main SDK package that bundles all instrumentation libraries
- **30+ instrumentation packages**: Individual OpenTelemetry instrumentations for LLM providers (OpenAI, Anthropic, Cohere, etc.), Vector DBs (Pinecone, Chroma, Qdrant, etc.), and frameworks (LangChain, LlamaIndex, etc.)
- **opentelemetry-semantic-conventions-ai**: Semantic conventions for AI/LLM telemetry
- **sample-app**: Example applications demonstrating SDK usage

The traceloop-sdk serves as the main entry point, importing and bundling all individual instrumentation packages via local path dependencies.

## Development Commands

### NX Monorepo Commands
The repository uses NX with Python plugin for managing the monorepo:

```bash
# List all projects
npx nx show projects

# Run specific target for a project
npx nx run traceloop-sdk:test
npx nx run traceloop-sdk:lint  
npx nx run traceloop-sdk:build

# Run command for all projects
npx nx run-many --target=test
npx nx run-many --target=lint
```

### Package-Specific Commands (run from packages/traceloop-sdk/)
```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest tests/
# Run single test file
poetry run pytest tests/test_sdk_initialization.py -v

# Linting
poetry run flake8

# Build package
poetry build

# Build for release (updates version references)
chmod +x ../../scripts/build-release.sh && ../../scripts/build-release.sh
```

### Working with Individual Instrumentation Packages
Each instrumentation package (e.g., opentelemetry-instrumentation-openai) follows the same pattern:
```bash
cd packages/opentelemetry-instrumentation-openai
poetry run pytest tests/
```

## Architecture

### Core SDK Structure
- `traceloop/sdk/__init__.py`: Main Traceloop class with init() method
- `traceloop/sdk/tracing/`: OpenTelemetry trace setup and management
- `traceloop/sdk/metrics/`: Metrics collection wrapper
- `traceloop/sdk/logging/`: Logging integration
- `traceloop/sdk/instruments.py`: Enum of available instrumentations
- `traceloop/sdk/client/`: Client for Traceloop cloud platform integration

### Instrumentation Pattern
All instrumentation packages follow OpenTelemetry's instrumentation pattern:
- `opentelemetry/instrumentation/{provider}/__init__.py`: Main instrumentor class
- Automatic patching of target library methods
- Span creation with AI semantic conventions
- Support for both sync and async operations

### Key Dependencies
- **OpenTelemetry**: Core tracing, metrics, and logging infrastructure
- **Poetry**: Python packaging and dependency management
- **Individual provider libraries**: OpenAI, Anthropic, etc. (test dependencies only)

## Testing Strategy

### Test Structure
- Unit tests for SDK core functionality
- Integration tests with VCR.py cassettes for reproducible API testing
- Async testing with pytest-asyncio
- Tests are organized by functionality (traces, metrics, privacy, etc.)

### VCR Testing
Many instrumentation packages use VCR.py to record/replay HTTP interactions:
```bash
# Re-record cassettes (requires real API keys)
poetry run pytest tests/test_openai.py --record-mode=rewrite
```

## Configuration

### Environment Variables
Key environment variables for testing and development:
- `TRACELOOP_API_KEY`: API key for Traceloop platform
- `TRACELOOP_BASE_URL`: Custom endpoint URL
- `TRACELOOP_TELEMETRY`: Enable/disable telemetry (default: true)
- Provider API keys for testing: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.

### SDK Initialization
```python
from traceloop.sdk import Traceloop

# Basic initialization
Traceloop.init()

# With custom configuration
Traceloop.init(
    app_name="my-app",
    disable_batch=True,  # For local development
    api_endpoint="custom-endpoint",
    instruments=set([Instruments.OPENAI, Instruments.ANTHROPIC])
)
```