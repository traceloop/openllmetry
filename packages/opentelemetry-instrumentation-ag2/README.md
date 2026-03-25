# OpenTelemetry AG2 Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-ag2/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-ag2.svg" alt="PyPI version">
</a>

This library allows tracing multi-agent workflows implemented with the [AG2 (formerly AutoGen) framework](https://github.com/ag2ai/ag2).

> **Note:** If you are using AG2's built-in OpenTelemetry instrumentation, disable it before using this package to avoid duplicate spans.

## Installation

```bash
pip install opentelemetry-instrumentation-ag2
```

## Example usage

```python
from opentelemetry.instrumentation.ag2 import AG2Instrumentor

AG2Instrumentor().instrument()
```

## Privacy

**By default, this instrumentation logs prompts, completions, and embeddings to span attributes**. This gives you a clear visibility into how your LLM application is working, and can make it easy to debug and evaluate the quality of the outputs.

However, you may want to disable this logging for privacy reasons, as they may contain highly sensitive data from your users. You may also simply want to reduce the size of your traces.

To disable logging, set the `TRACELOOP_TRACE_CONTENT` environment variable to `false`.

```bash
TRACELOOP_TRACE_CONTENT=false
```
