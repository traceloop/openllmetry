# OpenTelemetry Marqo Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-marqo/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-marqo.svg">
</a>

This library allows tracing client-side calls to Marqo vector DB sent with the official [Marqo library](https://github.com/marqo-ai/marqo).

## Installation

```bash
pip install opentelemetry-instrumentation-marqo
```

## Example usage

```python
from opentelemetry.instrumentation.marqo import MarqoInstrumentor

MarqoInstrumentor().instrument()
```
