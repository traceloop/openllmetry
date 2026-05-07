# OpenTelemetry Aleph Alpha Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-alephalpha/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-alephalpha.svg">
</a>

This library allows tracing calls to any of Aleph Alpha's endpoints sent with the official [Aleph Alpha Client](https://github.com/Aleph-Alpha/aleph-alpha-client).

## Quickstart

If you want to see tracing in action as quickly as possible, start with a tiny completion script:

```bash
pip install opentelemetry-instrumentation-alephalpha aleph-alpha-client
export AA_TOKEN=your_aleph_alpha_token
```

```python
from aleph_alpha_client import Client, CompletionRequest, Prompt
from opentelemetry.instrumentation.alephalpha import AlephAlphaInstrumentor

AlephAlphaInstrumentor().instrument()

client = Client(token="your_aleph_alpha_token", host="https://api.aleph-alpha.com")
request = CompletionRequest(
    prompt=Prompt.from_text("Tell me a joke about OpenTelemetry."),
    maximum_tokens=100,
)

response = client.complete(request, model="luminous-base")
print(response.completions[0].completion)
```

That is enough to create a traced completion call. If you want the prompt and completion text in span attributes, keep the default `TRACELOOP_TRACE_CONTENT=true`. If you want to suppress content tracing, set it to `false`.

## Installation

```bash
pip install opentelemetry-instrumentation-alephalpha
```

## Example usage

```python
from opentelemetry.instrumentation.alephalpha import AlephAlphaInstrumentor

AlephAlphaInstrumentor().instrument()
```

For a full request example, use the snippet above with `aleph_alpha_client.Client.complete(...)`.

## Privacy

**By default, this instrumentation logs prompts, completions, and embeddings to span attributes**. This gives you a clear visibility into how your LLM application is working, and can make it easy to debug and evaluate the quality of the outputs.

However, you may want to disable this logging for privacy reasons, as they may contain highly sensitive data from your users. You may also simply want to reduce the size of your traces.

To disable logging, set the `TRACELOOP_TRACE_CONTENT` environment variable to `false`.

```bash
TRACELOOP_TRACE_CONTENT=false
```
