"""Unit tests configuration module."""

import os

import pytest
from opentelemetry import metrics, trace
from opentelemetry.instrumentation.groq import GroqInstrumentor
from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    InMemoryMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from groq import Groq, AsyncGroq


@pytest.fixture(scope="session")
def exporter():
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    return exporter


@pytest.fixture(scope="session")
def reader():
    reader = InMemoryMetricReader(
        {Counter: AggregationTemporality.DELTA, Histogram: AggregationTemporality.DELTA}
    )
    return reader


@pytest.fixture(scope="session")
def meter_provider(reader):
    resource = Resource.create()
    meter_provider = MeterProvider(metric_readers=[reader], resource=resource)
    metrics.set_meter_provider(meter_provider)

    return meter_provider


@pytest.fixture(scope="session", autouse=True)
def instrument(exporter, reader, meter_provider):
    GroqInstrumentor(enrich_token_usage=True).instrument()

    yield

    exporter.shutdown()
    reader.shutdown()
    meter_provider.shutdown()


@pytest.fixture(autouse=True)
def clear_exporter_reader(exporter, reader):
    exporter.clear()
    reader.get_metrics_data()


@pytest.fixture(autouse=True)
def environment():
    os.environ["GROQ_API_KEY"] = "api-key"


@pytest.fixture
def groq_client():
    return Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )


@pytest.fixture
def async_groq_client():
    return AsyncGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization", "api-key"]}
