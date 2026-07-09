# OpenTelemetry Mistral AI Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-mistralai/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-mistralai.svg">
</a>

This library allows tracing calls to any of mistralai's endpoints sent with the official [Mistral AI library](https://github.com/mistralai-ai/mistralai-python).

## Installation

```bash
pip install opentelemetry-instrumentation-mistralai
```

## Example usage

```python
from opentelemetry.instrumentation.mistralai import MistralAiInstrumentor

MistralAiInstrumentor().instrument()
```

## Privacy

**By default, this instrumentation logs prompts, completions, and embeddings to span attributes**. This gives you a clear visibility into how your LLM application is working, and can make it easy to debug and evaluate the quality of the outputs.

However, you may want to disable this logging for privacy reasons, as they may contain highly sensitive data from your users. You may also simply want to reduce the size of your traces.

To disable logging, set the `TRACELOOP_TRACE_CONTENT` environment variable to `false`.

```bash
TRACELOOP_TRACE_CONTENT=false
```
