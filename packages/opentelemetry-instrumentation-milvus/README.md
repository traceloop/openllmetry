# OpenTelemetry Chroma Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-chromadb/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-chromadb.svg">
</a>

This library allows tracing client-side calls to Chroma vector DB sent with the official [Chroma library](https://github.com/chroma-core/chroma).

## Installation

```bash
pip install opentelemetry-instrumentation-chromadb
```

## Example usage

```python
from opentelemetry.instrumentation.chromadb import ChromaInstrumentor

ChromaInstrumentor().instrument()
```
