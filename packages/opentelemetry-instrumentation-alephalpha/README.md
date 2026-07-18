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

---

## Beginner-friendly walkthrough

This section walks you through tracing your first Aleph Alpha LLM call from scratch.

### Prerequisites

Install the required packages:

```bash
pip install opentelemetry-instrumentation-alephalpha \
            aleph_alpha_client \
            opentelemetry-sdk
```

You need an **Aleph Alpha API token**.  Sign up at <https://app.aleph-alpha.com> and create a token in your account settings.

### Step 1 – Set up OpenTelemetry

Before instrumenting anything you need an OpenTelemetry `TracerProvider` that knows where to export spans.  The simplest option for local development is to print spans to the console:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# Create a provider that prints spans to stdout
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
```

### Step 2 – Instrument the Aleph Alpha client

```python
from opentelemetry.instrumentation.alephalpha import AlephAlphaInstrumentor

AlephAlphaInstrumentor().instrument()
```

After this single call every subsequent Aleph Alpha API request is automatically wrapped in an OpenTelemetry span — no further code changes are needed.

### Step 3 – Make an LLM call

```python
import os
from aleph_alpha_client import Client, CompletionRequest, Prompt

client = Client(token=os.environ["AA_TOKEN"])

request = CompletionRequest(
    prompt=Prompt.from_text("Explain observability in one sentence."),
    maximum_tokens=100,
)
response = client.complete(request, model="luminous-base")
print(response.completions[0].completion)
```

### Complete minimal example

```python
"""
Minimal Aleph Alpha tracing example.

Requirements:
    pip install opentelemetry-instrumentation-alephalpha aleph_alpha_client opentelemetry-sdk

Environment:
    export AA_TOKEN=<your-aleph-alpha-api-token>
"""

import os

# 1. Set up OpenTelemetry (console exporter for local development)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# 2. Instrument the Aleph Alpha client
from opentelemetry.instrumentation.alephalpha import AlephAlphaInstrumentor

AlephAlphaInstrumentor().instrument()

# 3. Make a completion request — the span is captured automatically
from aleph_alpha_client import Client, CompletionRequest, Prompt

client = Client(token=os.environ["AA_TOKEN"])
request = CompletionRequest(
    prompt=Prompt.from_text("Explain observability in one sentence."),
    maximum_tokens=100,
)
response = client.complete(request, model="luminous-base")
print("Response:", response.completions[0].completion)
```

### Expected trace output

When you run the example above you will see a JSON span printed to the console.  The key attributes are:

| Attribute | Description | Example value |
|---|---|---|
| `gen_ai.system` | The LLM provider | `"AlephAlpha"` |
| `llm.request.type` | The type of LLM call | `"completion"` |
| `gen_ai.request.model` | The model used | `"luminous-base"` |
| `gen_ai.prompt.0.content` | The prompt text (if content tracing is enabled) | `"Explain observability…"` |
| `gen_ai.completion.0.content` | The model's reply (if content tracing is enabled) | `"Observability is…"` |
| `gen_ai.usage.input_tokens` | Number of prompt tokens consumed | `8` |
| `gen_ai.usage.output_tokens` | Number of generated tokens | `42` |
| `llm.usage.total_tokens` | Total tokens (input + output) | `50` |

### Using the Traceloop SDK (recommended)

For production use, the [Traceloop SDK](https://pypi.org/project/traceloop-sdk/) handles provider setup, instrumentation, and exporting for you:

```bash
pip install traceloop-sdk opentelemetry-instrumentation-alephalpha aleph_alpha_client
```

```python
import os
from aleph_alpha_client import Client, CompletionRequest, Prompt
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

# Initialise – instruments all supported LLM libraries automatically
Traceloop.init(app_name="my_alephalpha_app")

client = Client(token=os.environ["AA_TOKEN"])


@task(name="complete_prompt")
def complete_prompt(text: str) -> str:
    request = CompletionRequest(
        prompt=Prompt.from_text(text),
        maximum_tokens=100,
    )
    response = client.complete(request, model="luminous-base")
    return response.completions[0].completion


@workflow(name="main_workflow")
def main():
    answer = complete_prompt("Explain observability in one sentence.")
    print("Answer:", answer)


main()
```

A richer version of this example (with `@workflow` / `@task` decorators) can be found in [`packages/sample-app/sample_app/alephalpha_example.py`](../../sample-app/sample_app/alephalpha_example.py).

---

## Privacy

**By default, this instrumentation logs prompts, completions, and embeddings to span attributes**. This gives you a clear visibility into how your LLM application is working, and can make it easy to debug and evaluate the quality of the outputs.

However, you may want to disable this logging for privacy reasons, as they may contain highly sensitive data from your users. You may also simply want to reduce the size of your traces.

To disable logging, set the `TRACELOOP_TRACE_CONTENT` environment variable to `false`.

```bash
TRACELOOP_TRACE_CONTENT=false
```
