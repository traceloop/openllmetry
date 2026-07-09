# OpenTelemetry Cohere Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-cohere/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-cohere.svg">
</a>

This library allows tracing calls to any of Cohere's endpoints sent with the official [Cohere library](https://github.com/cohere-ai/cohere-python).

## Installation

```bash
pip install opentelemetry-instrumentation-cohere
```

## Example usage

```python
from opentelemetry.instrumentation.cohere import CohereInstrumentor

CohereInstrumentor().instrument()
```

## Privacy

**By default, this instrumentation logs prompts, completions, and embeddings to span attributes**. This gives you a clear visibility into how your LLM application is working, and can make it easy to debug and evaluate the quality of the outputs.

However, you may want to disable this logging for privacy reasons, as they may contain highly sensitive data from your users. You may also simply want to reduce the size of your traces.

To disable logging, set the `TRACELOOP_TRACE_CONTENT` environment variable to `false`.

```bash
TRACELOOP_TRACE_CONTENT=false
```
