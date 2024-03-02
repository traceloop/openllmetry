# OpenTelemetry CassIO Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-cassio/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-cassio.svg">
</a>

This library allows tracing client-side calls to Cassandra DB sent with [CassIO library](https://github.com/CassioML/cassio).

## Installation

```bash
pip install opentelemetry-instrumentation-cassio
```

## Example usage

```python
from opentelemetry.instrumentation.cassio import CassIOInstrumentor

CassIOInstrumentor().instrument()
```
