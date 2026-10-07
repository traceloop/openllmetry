"""Beginner-friendly Aleph Alpha completion tracing example."""

from __future__ import annotations

import os
import sys

from aleph_alpha_client import Client, CompletionRequest, Prompt
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.trace import set_tracer_provider

from opentelemetry.instrumentation.alephalpha import AlephAlphaInstrumentor


def configure_tracing() -> None:
    """Send completed spans to stdout so the example needs no collector."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    set_tracer_provider(provider)
    AlephAlphaInstrumentor().instrument(tracer_provider=provider)


def main() -> int:
    token = os.getenv("AA_TOKEN")
    if not token:
        print("Set AA_TOKEN before running this example.", file=sys.stderr)
        print("Example: set AA_TOKEN=your-token", file=sys.stderr)
        return 2

    configure_tracing()
    client = Client(token=token)
    request = CompletionRequest(
        prompt=Prompt.from_text("Explain OpenTelemetry tracing in one sentence."),
        maximum_tokens=64,
    )
    response = client.complete(request, model=os.getenv("AA_MODEL", "luminous-base"))
    print(response.completions[0].completion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
