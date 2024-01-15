# OpenTelemetry VertexAI Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-vertexai/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-vertexai.svg">
</a>

This library allows tracing VertexAI prompts and completions sent with the official [VertexAI library](https://github.com/googleapis/python-aiplatform).

## Installation

```bash
pip install opentelemetry-instrumentation-vertexai
```

## Privacy

**By default, this instrumentation logs prompts, completions, and embeddings to span attributes**. This gives you a clear visibility into how your LLM application is working, and can make it easy to debug and evaluate the quality of the outputs.

However, you may want to disable this logging for privacy reasons, as they may contain highly sensitive data from your users. You may also simply want to reduce the size of your traces.

To disable logging, set the `TRACELOOP_TRACE_CONTENT` environment variable to `false`.

```bash
TRACELOOP_TRACE_CONTENT=false
```
