"""Unit tests configuration module."""

import os
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.instrumentation.alephalpha import AlephAlphaInstrumentor

pytest_plugins = ["pytest_recording"]


@pytest.fixture(scope="session")
def exporter():
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    AlephAlphaInstrumentor().instrument()

    return exporter


@pytest.fixture(autouse=True)
def clear_exporter(exporter):
    exporter.clear()


@pytest.fixture(autouse=True)
def environment():
    if "AA_TOKEN" not in os.environ:
        os.environ["AA_TOKEN"] = ("test_api_key")


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization", "Authorization"],
        "decode_compressed_response": True,
        "record_mode": os.getenv("VCR_RECORD_MODE", "none"),
        "filter_post_data_parameters": ["token"],
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
    }
