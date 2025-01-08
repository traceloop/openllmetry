"""Unit tests configuration module."""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.instrumentation.google_generativeai import (
    GoogleGenerativeAiInstrumentor,
)

pytest_plugins = []

def pytest_sessionstart(session):
    """
    Pytest hook that runs at the start of the test session.
    Instruments the Google Generative AI library.
    """
    GoogleGenerativeAiInstrumentor().instrument()

@pytest.fixture(scope="session")
def exporter():
    """
    Fixture that creates an InMemorySpanExporter and a TracerProvider
    configured to use it. It sets the global TracerProvider.
    """
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    return exporter

@pytest.fixture(autouse=True)
def clear_exporter(exporter):
    """
    Fixture that automatically clears the spans from the exporter
    before each test.
    """
    exporter.clear()

@pytest.fixture(scope="module")
def vcr_config():
    """
    VCR configuration fixture.
    """
    return {"filter_headers": ["authorization"]}