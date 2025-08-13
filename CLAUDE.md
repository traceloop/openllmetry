# OpenLLMetry Repository Guide

## Repository Structure
This repository contains multiple PyPI-publishable packages organized and orchestrated using Nx workspace management.

## Package Management
All packages use Poetry as the package manager. Always execute commands through Poetry:
```bash
poetry run <command>
```

## Testing with VCR Cassettes
Tests utilize recordings/cassettes for API calls. When making changes that affect API interactions, re-record VCR cassettes to ensure test accuracy. Creating new cassettes typically requires API keys (OpenAI, Anthropic, etc.) - request these from the user when needed.

## Semantic Conventions
The semantic convention package follows the OpenTelemetry GenAI specification:
https://opentelemetry.io/docs/specs/semconv/gen-ai/

## Instrumentation Packages
Instrumentation packages should leverage the semantic conventions package. Their purpose is to instrument AI-related libraries and generate spans and tracing data compliant with OpenTelemetry semantic conventions.

## Code Quality
Flake8 is used for code linting and formatting.