"""Unit tests configuration module."""

import os
import pytest
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.pinecone import PineconeInstrumentor

pytest_plugins = []


@pytest.fixture(scope="session")
def traces_exporter():
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    yield exporter

    exporter.clear()


@pytest.fixture(scope="session")
def metrics_reader():
    resource = Resource.create()
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader], resource=resource)

    metrics.set_meter_provider(provider)

    return reader


@pytest.fixture(autouse=True, scope="session")
def instrument(traces_exporter, metrics_reader):
    OpenAIInstrumentor().instrument()
    PineconeInstrumentor().instrument()


@pytest.fixture(autouse=True)
def environment():
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "test_api_key"
    if "PINECONE_API_KEY" not in os.environ:
        os.environ["PINECONE_API_KEY"] = "test_api_key"
    if "PINECONE_ENVIRONMENT" not in os.environ:
        os.environ["PINECONE_ENVIRONMENT"] = "gcp-starter"


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization"]}
