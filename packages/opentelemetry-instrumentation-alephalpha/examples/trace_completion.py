"""Trace one Aleph Alpha completion and print the span locally."""

import os

from aleph_alpha_client import Client, CompletionRequest, Prompt
from opentelemetry import trace
from opentelemetry.instrumentation.alephalpha import AlephAlphaInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor


def main() -> None:
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
