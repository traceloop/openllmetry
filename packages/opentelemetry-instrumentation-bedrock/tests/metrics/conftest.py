"""Unit tests configuration module."""

import pytest
from opentelemetry import metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.instrumentation.bedrock import BedrockInstrumentor

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture(scope="session")
def metrics_test_context():
    resource = Resource.create()
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader], resource=resource)

    metrics.set_meter_provider(provider)

    # Without the following lines, span.is_recording() is False
    # so that _handle_call and _handle_stream_call will be skipped
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(processor)
    trace.set_tracer_provider(trace_provider)

    BedrockInstrumentor(enrich_token_usage=True).instrument()

    return provider, reader


@pytest.fixture(scope="session", autouse=True)
def clear_metrics_test_context(metrics_test_context):
    provider, reader = metrics_test_context

    reader.shutdown()
    provider.shutdown()
