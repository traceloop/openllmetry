"""Unit tests configuration module."""

import os

import pytest
import voyageai
from opentelemetry.instrumentation.voyageai import VoyageAIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytest_plugins = []


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture
def voyageai_client():
    return voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))


@pytest.fixture
def async_voyageai_client():
    return voyageai.AsyncClient(api_key=os.environ.get("VOYAGE_API_KEY"))


@pytest.fixture(scope="function")
def instrument_legacy(tracer_provider):
    instrumentor = VoyageAIInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
    )

    yield instrumentor

    instrumentor.uninstrument()


@pytest.fixture(autouse=True)
def environment():
    if "VOYAGE_API_KEY" not in os.environ:
        os.environ["VOYAGE_API_KEY"] = "test_api_key"


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization", "x-voyage-api-key"]}
