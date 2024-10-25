"""Unit tests configuration module."""

import os
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.instrumentation.alephalpha import AlephAlphaInstrumentor

pytest_plugins = []

@pytest.fixture(scope="session")
def exporter():
    """Fixture for creating an in-memory span exporter for testing."""
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    # Instrument the Aleph Alpha client
    AlephAlphaInstrumentor().instrument()

    return exporter

@pytest.fixture(autouse=True)
def clear_exporter(exporter):
    """Automatically clear the exporter before each test."""
    exporter.clear()

@pytest.fixture(autouse=True)
def environment():
    """Set up the environment variable for the test."""
    if "AA_TOKEN" not in os.environ:
        os.environ["AA_TOKEN"] = "test_api_key"  # Replace with a real test API key if needed

@pytest.fixture(scope="module")
def vcr_config():
    """Configuration for VCR (if used) to filter headers and decode responses."""
    return {"filter_headers": ["authorization"], "decode_compressed_response": True}
