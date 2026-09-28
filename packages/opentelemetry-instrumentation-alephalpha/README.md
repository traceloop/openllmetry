# OpenTelemetry Aleph Alpha Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-alephalpha/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-alephalpha.svg">
</a>

This library allows tracing calls to any of Aleph Alpha's endpoints sent with the official [Aleph Alpha Client](https://github.com/Aleph-Alpha/aleph-alpha-client).

## Installation

```bash
pip install opentelemetry-instrumentation-alephalpha
```

## Example usage

```python
from opentelemetry.instrumentation.alephalpha import AlephAlphaInstrumentor

AlephAlphaInstrumentor().instrument()
```

## Beginner example

The following example makes one completion request and prints the trace to the
terminal. It is useful for understanding the complete flow locally before
connecting the application to an observability backend.

Install the instrumentation package and the Aleph Alpha client:

```bash
pip install opentelemetry-instrumentation-alephalpha aleph-alpha-client opentelemetry-sdk
```

Set your Aleph Alpha API token:

```bash
export AA_TOKEN="your-api-token"
```

Create `trace_completion.py`:

```python
import os

from aleph_alpha_client import Client, CompletionRequest, Prompt
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.instrumentation.alephalpha import AlephAlphaInstrumentor


def main():
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    AlephAlphaInstrumentor().instrument(tracer_provider=tracer_provider)

    client = Client(token=os.environ["AA_TOKEN"])
    request = CompletionRequest(
        prompt=Prompt.from_text("Explain ETL in one sentence."),
        maximum_tokens=100,
    )
    response = client.complete(request, model="luminous-base")

    print(response.completions[0].completion)


if __name__ == "__main__":
    main()
```

Run it with:

```bash
python trace_completion.py
```

The terminal prints the model response followed by an OpenTelemetry span. The
span contains information such as the model name, request type, duration, and
token usage. Prompt and completion content are also recorded by default; see
the [Privacy](#privacy) section if your application handles sensitive data.

The example demonstrates the four steps involved in tracing an LLM call:

1. Create an OpenTelemetry tracer provider.
2. Add an exporter that prints completed spans.
3. Instrument the Aleph Alpha client.
4. Make a normal client request. The instrumentation creates the span.

## Privacy

**By default, this instrumentation logs prompts, completions, and embeddings to span attributes**. This gives you a clear visibility into how your LLM application is working, and can make it easy to debug and evaluate the quality of the outputs.

However, you may want to disable this logging for privacy reasons, as they may contain highly sensitive data from your users. You may also simply want to reduce the size of your traces.

To disable logging, set the `TRACELOOP_TRACE_CONTENT` environment variable to `false`.

```bash
TRACELOOP_TRACE_CONTENT=false
```
