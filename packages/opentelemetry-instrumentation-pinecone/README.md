# OpenTelemetry Pinecone Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-pinecone/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-pinecone.svg">
</a>

This library allows tracing client-side calls to Pinecone vector DB sent with the official [Pinecone library](https://github.com/pinecone-io/pinecone-python-client).

## Installation

```bash
pip install opentelemetry-instrumentation-pinecone
```

## Example usage

```python
from opentelemetry.instrumentation.pinecone import PineconeInstrumentor

PineconeInstrumentor().instrument()
```
