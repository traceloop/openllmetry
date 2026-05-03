"""Beginner-friendly example: trace a single LLM call with Traceloop + Ollama.

This example is meant for first-time users of OpenLLMetry. It shows the smallest
possible setup that produces a real trace so you can see what observability
actually looks like.

Why Ollama? It runs locally — no API key, no signup, no cost.

Prerequisites
-------------
1. Install Ollama: https://ollama.com/download
2. Pull a small model:
       ollama pull llama3.2
3. Make sure the daemon is running (`ollama serve` if it isn't already).

Run it
------
    uv run python sample_app/ollama_beginner_example.py

What you should see
-------------------
The model's reply, followed by one or more JSON span objects printed to your
terminal. Each span describes a unit of work — the LLM call, its inputs, its
outputs, and how long it took. That is the trace.
"""

from ollama import chat
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

# Step 1 — Initialize Traceloop.
#
# `exporter=ConsoleSpanExporter()` sends spans to your terminal instead of to
# Traceloop's cloud dashboard. Great for learning: you see the trace
# immediately, with no account or API key.
#
# `disable_batch=True` flushes spans right after each call, so you don't have
# to wait for a batch to fill up before you see output.
Traceloop.init(
    app_name="ollama-beginner-example",
    exporter=ConsoleSpanExporter(),
    disable_batch=True,
)


# Step 2 — Wrap your logic in a `@workflow`.
#
# A workflow is a logical grouping of operations. Traceloop creates a parent
# span for the workflow, and any LLM calls inside it become child spans. This
# is what lets you see the full request/response in one trace later.
@workflow(name="ollama_greeting")
def greet():
    # Step 3 — Make any LLM call. Traceloop auto-instruments the Ollama client,
    # so this single call produces a span with the prompt, response, model
    # name, and token usage — without any extra code from you.
    response = chat(
        model="llama3.2",
        messages=[
            {"role": "user", "content": "Say hello in one sentence."},
        ],
    )
    print(response.message.content)
    return response


if __name__ == "__main__":
    greet()
