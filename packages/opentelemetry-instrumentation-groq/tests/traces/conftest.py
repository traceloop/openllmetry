"""Unit tests configuration module."""

import os

import pytest
from groq import AsyncGroq, Groq
from opentelemetry.instrumentation.groq import GroqInstrumentor
from opentelemetry.instrumentation.groq.utils import TRACELOOP_TRACE_CONTENT
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    InMemoryMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function", name="log_exporter")
def fixture_log_exporter():
    exporter = InMemoryLogExporter()
    yield exporter


@pytest.fixture(scope="function", name="logger_provider")
def fixture_logger_provider(log_exporter):
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
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


@pytest.fixture(scope="function")
def instrument_legacy(reader, tracer_provider, meter_provider):
    instrumentor = GroqInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor

    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_content(
    reader, tracer_provider, logger_provider, meter_provider
):
    os.environ.update({TRACELOOP_TRACE_CONTENT: "True"})

    instrumentor = GroqInstrumentor(
        use_legacy_attributes=False,
    )
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor

    os.environ.pop(TRACELOOP_TRACE_CONTENT, None)
    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_no_content(
    reader, tracer_provider, logger_provider, meter_provider
):
    os.environ.update({TRACELOOP_TRACE_CONTENT: "False"})

    instrumentor = GroqInstrumentor(
        use_legacy_attributes=False
    )
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor

    os.environ.pop(TRACELOOP_TRACE_CONTENT, None)
    instrumentor.uninstrument()


@pytest.fixture(autouse=True)
def environment():
    if not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = "api-key"


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization", "api-key"]}
