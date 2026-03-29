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

## Supported Embedding Models

Token usage is captured for Cohere text embedding models, including:

- `embed-english-v3.0`
- `embed-english-light-v3.0`
- `embed-multilingual-v3.0`
- `embed-multilingual-light-v3.0`

The following span attributes are set for embedding calls (attribute names follow the project's semantic conventions -- `gen_ai.*` for the newer spec, `llm.*` for legacy attributes not yet migrated):

- `gen_ai.usage.prompt_tokens` -- Number of input tokens
- `gen_ai.usage.completion_tokens` -- Number of output tokens (0 for embeddings)
- `llm.usage.total_tokens` -- Total token count

## Privacy

**By default, this instrumentation logs prompts, completions, and embeddings to span attributes**. This gives you a clear visibility into how your LLM application is working, and can make it easy to debug and evaluate the quality of the outputs.

However, you may want to disable this logging for privacy reasons, as they may contain highly sensitive data from your users. You may also simply want to reduce the size of your traces.

To disable logging, set the `TRACELOOP_TRACE_CONTENT` environment variable to `false`.

```bash
TRACELOOP_TRACE_CONTENT=false
```
