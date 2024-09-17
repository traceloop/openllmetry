"""Unit tests configuration module."""

import pytest
from opentelemetry import trace
from opentelemetry.instrumentation.bedrock import BedrockInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture(scope="session")
def exporter():
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    return exporter


@pytest.fixture(scope="session", autouse=True)
def instrument(exporter):
    BedrockInstrumentor(enrich_token_usage=True).instrument()

    yield

    exporter.shutdown()


@pytest.fixture(autouse=True)
def clear_exporter(exporter):
    exporter.clear()
