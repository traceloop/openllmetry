"""Unit tests configuration module."""

import os
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.instrumentation.haystack import HaystackInstrumentor

pytest_plugins = []


@pytest.fixture(scope="session")
def exporter():
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    HaystackInstrumentor().instrument()

    return exporter


@pytest.fixture(autouse=True)
def environment():
    os.environ["OPENAI_API_KEY"] = "test_api_key"
    os.environ["TRACELOOP_TRACE_CONTENT"] = "true"


@pytest.fixture(autouse=True)
def clear_exporter(exporter):
    exporter.clear()


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization"]}
