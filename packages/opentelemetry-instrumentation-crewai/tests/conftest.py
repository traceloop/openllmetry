"""Unit tests configuration module."""

import pytest
from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    InMemoryMetricReader,
)
from opentelemetry.sdk.resources import Resource
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


@pytest.fixture(scope="function", name="reader")
def fixture_reader():
    reader = InMemoryMetricReader(
        {Counter: AggregationTemporality.DELTA, Histogram: AggregationTemporality.DELTA}
    )
    return reader


@pytest.fixture(scope="function", name="meter_provider")
def fixture_meter_provider(reader):
    resource = Resource.create()
    meter_provider = MeterProvider(metric_readers=[reader], resource=resource)
    return meter_provider


@pytest.fixture(scope="function")
def instrument(reader, tracer_provider, meter_provider):
    """Real instrumentation against an in-memory exporter.

    BaseInstrumentor is a singleton (its __new__ returns a shared instance),
    so any instance-attribute mocks left behind by other tests must be
    removed before the real methods can run again.
    """
    from opentelemetry.instrumentation.crewai import CrewAIInstrumentor

    instrumentor = CrewAIInstrumentor()
    # Restore real methods in case a previous test mocked them on the singleton.
    instrumentor.__dict__.pop("instrument", None)
    instrumentor.__dict__.pop("uninstrument", None)
    if instrumentor._is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()

    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor

    instrumentor.uninstrument()
