# OpenTelemetry Agno Instrumentation

This library provides automatic instrumentation for the [Agno](https://github.com/agno-agi/agno) framework.

## Installation

```bash
pip install opentelemetry-instrumentation-agno
```

## Usage

```python
from opentelemetry.instrumentation.agno import AgnoInstrumentor

AgnoInstrumentor().instrument()
```

## Supported Features

This instrumentation captures:
- Agent execution (sync and async)
- Team operations
- Model invocations
- Function calls
- Streaming responses

## Links

- [Agno Framework](https://github.com/agno-agi/agno)
- [OpenTelemetry](https://opentelemetry.io/)
