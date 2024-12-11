# OpenTelemetry Google Generative AI Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-google-generativeai/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-google-generativeai.svg">
</a>

This library allows tracing Google Gemini prompts and completions sent with the official [Google Generative AI library](https://github.com/google-gemini/generative-ai-python).

## Installation

```bash
pip install opentelemetry-instrumentation-google-generativeai
```

## Example usage

```python
from opentelemetry.instrumentation.google_generativeai import GoogleGenerativeAiInstrumentor

GoogleGenerativeAiInstrumentor().instrument()
```

## Privacy

**By default, this instrumentation logs prompts, completions, and embeddings to span attributes**. This gives you a clear visibility into how your LLM application is working, and can make it easy to debug and evaluate the quality of the outputs.

However, you may want to disable this logging for privacy reasons, as they may contain highly sensitive data from your users. You may also simply want to reduce the size of your traces.

To disable logging, set the `TRACELOOP_TRACE_CONTENT` environment variable to `false`.

```bash
TRACELOOP_TRACE_CONTENT=false
```
