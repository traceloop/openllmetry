"""
Example demonstrating how to use OpenTelemetry metrics with GenAI bucket boundaries.

This example shows how to set up metrics with boundaries
for GenAI applications using opentelemetry-semantic-conventions-ai.

BACKGROUND:
This example has been modified to use standalone package instrumentation approach
instead of manual meter setup. This change was made because we are waiting for
the new semantic conventions release before adding comprehensive unit tests.
For now, this example serves as a way to verify the correctness of bucket
boundaries by demonstrating their usage in a real instrumentation scenario.
"""

import os
from openai import OpenAI
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter


otlp_endpoint = os.getenv("OTLP_ENDPOINT", "http://localhost:4317")

metric_reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint=otlp_endpoint),
    export_interval_millis=1000
)

metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

trace_provider = TracerProvider()
trace.set_tracer_provider(trace_provider)

# Set up trace exporter
trace_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
trace_provider.add_span_processor(BatchSpanProcessor(trace_exporter))

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

OpenAIInstrumentor().instrument()


def test_openai_operations():
    """Test OpenAI operations to verify bucket boundaries in collector."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Tell me a short story about OpenTelemetry"}],
            max_tokens=1024
        )

        print(f"Response: {response.choices[0].message.content}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    """check metric histogram buckets in opentelemetry collector"""
    test_openai_operations()
