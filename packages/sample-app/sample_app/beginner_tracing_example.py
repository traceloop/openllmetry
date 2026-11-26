"""Minimal walkthrough: trace an LLM call with OpenLLMetry.

Three pieces, in the order you meet them in real code:

  Traceloop.init  — wire up the SDK so spans are produced
  @workflow       — top-level unit of work (one user-visible action)
  @task           — sub-step nested under that workflow

Uses OpenAI here, but the structure is the same for any provider this
repo instruments.

    export OPENAI_API_KEY=sk-...
    uv run python sample_app/beginner_tracing_example.py

This example wires a ConsoleSpanExporter so spans land on stdout out of
the box. For real backends (Traceloop, Jaeger, Honeycomb, Datadog, …)
set `TRACELOOP_API_KEY` or the standard `OTEL_EXPORTER_OTLP_*` env vars
and drop the explicit exporter argument.
"""

from openai import OpenAI
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

# `app_name` shows up on every span. `disable_batch=True` flushes
# synchronously, which is easier to read in a short script — leave the
# default on in production so flushing doesn't block your hot path.
# The explicit `exporter` makes spans visible on stdout even when no
# TRACELOOP_API_KEY or OTLP endpoint is configured.
Traceloop.init(
    app_name="beginner-tracing-example",
    disable_batch=True,
    exporter=ConsoleSpanExporter(),
)

client = OpenAI()


@task(name="tell_joke")
def tell_joke() -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Tell me a short joke about telemetry."}],
    )
    return response.choices[0].message.content or ""


@task(name="explain_joke")
def explain_joke(joke: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"Explain the joke in one sentence:\n\n{joke}"},
        ],
    )
    return response.choices[0].message.content or ""


# Everything called inside the workflow — including the two @task functions
# above — gets attached to the same trace automatically.
@workflow(name="joke_and_explanation")
def run() -> None:
    joke = tell_joke()
    print("Joke:", joke)

    explanation = explain_joke(joke)
    print("Explanation:", explanation)


if __name__ == "__main__":
    run()
