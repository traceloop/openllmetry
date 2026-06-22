# OpenTelemetry DeepSeek Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-deepseek/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-deepseek.svg" alt="PyPI version">
</a>

This library allows tracing prompts and completions sent to [DeepSeek](https://platform.deepseek.com/) using the official [OpenAI SDK](https://github.com/openai/openai-python), since DeepSeek's API is OpenAI-compatible.

## Installation

```bash
pip install opentelemetry-instrumentation-deepseek
```

## Example usage

```python
from opentelemetry.instrumentation.deepseek import DeepSeekInstrumentor

DeepSeekInstrumentor().instrument()
```

This instrumentor patches `openai.resources.chat.completions.Completions.create` and `AsyncCompletions.create`, but only creates spans for requests made through a client configured with a DeepSeek `base_url`. Regular OpenAI clients are left untouched, so this instrumentation can be safely enabled alongside [opentelemetry-instrumentation-openai](https://pypi.org/project/opentelemetry-instrumentation-openai/).

```python
from openai import OpenAI

client = OpenAI(
    api_key="<DEEPSEEK_API_KEY>",
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## DeepSeek-R1 reasoning content

The `deepseek-reasoner` model (DeepSeek-R1) returns its chain-of-thought in a `reasoning_content` field on the response message, in addition to the final `content`. This instrumentation captures `reasoning_content` as the `gen_ai.deepseek.reasoning_content` span attribute, for both streaming and non-streaming responses.

## Privacy

**By default, this instrumentation logs prompts, completions, and embeddings to span attributes**. This gives you a clear visibility into how your LLM application is working, and can make it easy to debug and evaluate the quality of the outputs.

However, you may want to disable this logging for privacy reasons, as they may contain highly sensitive data from your users. You may also simply want to reduce the size of your traces.

To disable logging, set the `TRACELOOP_TRACE_CONTENT` environment variable to `false`.

```bash
TRACELOOP_TRACE_CONTENT=false
```
