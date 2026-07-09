# OpenTelemetry Qdrant Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-qdrant/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-qdrant.svg">
</a>

This library allows tracing client-side calls to Qdrant vector DB sent with the official [Qdrant client library](https://github.com/qdrant/qdrant-client).

## Installation

```bash
pip install opentelemetry-instrumentation-qdrant
```

## Example usage

```python
from opentelemetry.instrumentation.qdrant import QdrantInstrumentor

QdrantInstrumentor().instrument()
```
