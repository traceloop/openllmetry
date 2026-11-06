"""
Beginner-friendly demo: trace an Aleph Alpha LLM call with OpenTelemetry.

What it shows
-------------
1. Setting up an OpenTelemetry TracerProvider (the thing that collects spans).
2. Turning on the Aleph Alpha instrumentation with ONE line:
       AlephAlphaInstrumentor().instrument(tracer_provider=provider)
3. Making a normal `client.complete(...)` call - no tracing code around it.
4. Printing the captured trace: a parent span from our app, and a child
   `alephalpha.completion` span created automatically by the instrumentation,
   carrying the model, prompt, completion and token usage.

How to run (from the repo root)
-------------------------------
    cd packages/sample-app
    uv sync --all-groups
    uv run python sample_app/alephalpha_tracing_demo.py

    # Ask your own question
    uv run python sample_app/alephalpha_tracing_demo.py "What is a trace?"

    # Also dump every span as raw OpenTelemetry JSON
    uv run python sample_app/alephalpha_tracing_demo.py --show-raw

No API key? No problem. If AA_TOKEN is not set (or --offline is passed) the
demo answers from a fake HTTP server built into this file. The real Aleph Alpha
client and the real instrumentation still run - only the network is faked.
To call the real API:  set AA_TOKEN=<your token>  (PowerShell: $env:AA_TOKEN="...")
"""

import argparse
import json
import os

import requests
from aleph_alpha_client import Client, CompletionRequest, Prompt
from opentelemetry import trace
from opentelemetry.instrumentation.alephalpha import AlephAlphaInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

DEFAULT_QUESTION = "Explain OpenTelemetry to a five year old in two sentences."


# ---------------------------------------------------------------------------
# Step 1: set up tracing
# ---------------------------------------------------------------------------
def setup_tracing(show_raw: bool) -> InMemorySpanExporter:
    """Create a TracerProvider and turn on the Aleph Alpha instrumentation."""
    provider = TracerProvider(resource=Resource.create({"service.name": "alephalpha-tracing-demo"}))

    # Keep finished spans in memory so we can print a friendly summary at the end.
    memory_exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory_exporter))

    # Optionally print every span as raw JSON, exactly as OpenTelemetry sees it.
    if show_raw:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)

    # The one line that matters: every Client.complete() call now creates a span.
    AlephAlphaInstrumentor().instrument(tracer_provider=provider)

    return memory_exporter


# ---------------------------------------------------------------------------
# Step 2: an ordinary LLM application
# ---------------------------------------------------------------------------
tracer = trace.get_tracer("alephalpha-tracing-demo")


def ask(client: Client, question: str, model: str) -> str:
    # A span of our own, so the LLM span shows up as its child in the trace tree.
    with tracer.start_as_current_span("ask_question") as span:
        span.set_attribute("app.question.length", len(question))

        request = CompletionRequest(
            prompt=Prompt.from_text(f"Q: {question}\nA:"),
            maximum_tokens=64,
            stop_sequences=["\nQ:"],
        )
        # Pass `model` as a keyword argument: the instrumentation reads it from
        # kwargs to fill in the `gen_ai.request.model` attribute.
        response = client.complete(request, model=model)

        answer = response.completions[0].completion.strip()
        span.set_attribute("app.answer.length", len(answer))
        return answer


# ---------------------------------------------------------------------------
# Step 3: show what was captured
# ---------------------------------------------------------------------------
def print_trace_summary(spans) -> None:
    print("\n=== Captured trace ===")
    children = {}
    for s in spans:
        parent_id = s.parent.span_id if s.parent else None
        children.setdefault(parent_id, []).append(s)

    def show(span, depth):
        indent = "    " * depth
        duration_ms = (span.end_time - span.start_time) / 1_000_000
        status = span.status.status_code.name
        print(f"{indent}- {span.name}  ({duration_ms:.1f} ms, kind={span.kind.name}, status={status})")
        for key, value in sorted(span.attributes.items()):
            text = str(value).replace("\n", " ")
            if len(text) > 80:
                text = text[:77] + "..."
            print(f"{indent}    {key} = {text}")
        for child in sorted(children.get(span.context.span_id, []), key=lambda c: c.start_time):
            show(child, depth + 1)

    for root in children.get(None, []):
        print(f"trace_id = {root.context.trace_id:032x}")
        show(root, 0)


# ---------------------------------------------------------------------------
# Offline mode: a tiny fake Aleph Alpha server
# ---------------------------------------------------------------------------
class FakeAlephAlphaAdapter(requests.adapters.BaseAdapter):
    """Answers HTTP requests locally with a canned /complete response."""

    def send(self, request, **kwargs):
        body = {
            "model_version": "offline-demo",
            "num_tokens_prompt_total": 18,
            "num_tokens_generated": 27,
            "completions": [
                {
                    "completion": " OpenTelemetry is like a diary for computer programs. It writes down "
                    "what they did and how long it took, so grown-ups can fix them when they get slow.",
                    "finish_reason": "stop_sequence_reached",
                }
            ],
        }
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(body).encode()
        response.headers["Content-Type"] = "application/json"
        response.url = request.url
        response.request = request
        return response

    def close(self):
        pass


def make_client(offline: bool, host: str) -> Client:
    if offline:
        client = Client(token="offline-demo-token", host=host)
        client.session.mount("https://", FakeAlephAlphaAdapter())
        client.session.mount("http://", FakeAlephAlphaAdapter())
        return client
    return Client(token=os.environ["AA_TOKEN"], host=host)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace an Aleph Alpha LLM call with OpenTelemetry.")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--model", default="luminous-base", help="Aleph Alpha model name")
    parser.add_argument("--host", default="https://api.aleph-alpha.com", help="Aleph Alpha API host")
    parser.add_argument("--offline", action="store_true", help="use the built-in fake server")
    parser.add_argument("--show-raw", action="store_true", help="also print raw span JSON")
    args = parser.parse_args()

    offline = args.offline or not os.getenv("AA_TOKEN")
    memory_exporter = setup_tracing(args.show_raw)
    client = make_client(offline, args.host)

    print(f"Mode:     {'offline (fake server)' if offline else 'live API'}")
    print(f"Model:    {args.model}")
    print(f"Question: {args.question}")
    answer = ask(client, args.question, args.model)
    print(f"Answer:   {answer}")

    print_trace_summary(memory_exporter.get_finished_spans())


if __name__ == "__main__":
    main()
