# OpenTelemetry LanceDB Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-lancedb/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-lancedb.svg">
</a>

This library allows tracing client-side calls to LanceDB sent with the official [LanceDB library](https://github.com/lancedb/lancedb).

## Installation

```bash
pip install opentelemetry-instrumentation-lancedb
```

## Example usage

```python
from opentelemetry.instrumentation.lancedb import LanceInstrumentor

LanceInstrumentor().instrument()
```
