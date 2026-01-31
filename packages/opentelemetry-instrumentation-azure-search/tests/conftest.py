"""Unit tests configuration module."""

import os
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor


@pytest.fixture(scope="session", autouse=True)
def environment():
    """Set up environment variables for testing."""
    if not os.getenv("AZURE_SEARCH_ENDPOINT"):
        os.environ["AZURE_SEARCH_ENDPOINT"] = "https://traceloop-otel-os.search.windows.net"
    if not os.getenv("AZURE_SEARCH_ADMIN_KEY"):
        os.environ["AZURE_SEARCH_ADMIN_KEY"] = "test-api-key"
    if not os.getenv("AZURE_SEARCH_INDEX_NAME"):
        os.environ["AZURE_SEARCH_INDEX_NAME"] = "test-hotels"


@pytest.fixture(scope="module")
def vcr_config():
    """VCR configuration for recording/replaying API calls."""
    return {
        "filter_headers": [
            "api-key",
            "authorization",
            "User-Agent",
            "x-ms-client-request-id",
        ],
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
        "allow_playback_repeats": True,
    }


@pytest.fixture(scope="session")
def exporter():
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    AzureSearchInstrumentor().instrument()

    return exporter


@pytest.fixture(autouse=True)
def clear_exporter(exporter):
    exporter.clear()
