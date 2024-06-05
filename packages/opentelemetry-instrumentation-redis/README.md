# OpenTelemetry Redis Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-redis/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-redis.svg">
</a>

This library allows tracing client-side calls to Redis vector DB sent with the official [Redis client library](https://github.com/redis/redis).

## Installation

```bash
pip install opentelemetry-instrumentation-redis
```

## Example usage

```python
from opentelemetry.instrumentation.redis import RedisInstrumentor

RedisInstrumentor().instrument()
```
