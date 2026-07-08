"""Unit tests configuration module."""

import os

from google import genai
import pytest
from opentelemetry.instrumentation.google_generativeai import (
    GoogleGenerativeAiInstrumentor,
)
from opentelemetry.instrumentation.google_generativeai.utils import (
    TRACELOOP_TRACE_CONTENT,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import set_tracer_provider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader


pytest_plugins = []


@pytest.fixture(scope="session")
def exporter(metrics_test_context):
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    set_tracer_provider(provider)

    meter_provider, _ = metrics_test_context
    GoogleGenerativeAiInstrumentor().instrument(meter_provider=meter_provider)

    return exporter


@pytest.fixture(scope="function")
def tracer_provider():
    provider = TracerProvider()
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


@pytest.fixture
def genai_client():
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return client


@pytest.fixture(scope="function")
def instrument_legacy(tracer_provider):
    instrumentor = GoogleGenerativeAiInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
    )

    yield instrumentor

    instrumentor.uninstrument()


@pytest.fixture(scope="session")
def metrics_test_context():
    resource = Resource.create()
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader], resource=resource)
    metrics.set_meter_provider(provider)
    return provider, reader


@pytest.fixture(scope="session", autouse=True)
def clear_metrics_test_context(metrics_test_context):
    yield
    provider, reader = metrics_test_context
    reader.shutdown()
    provider.shutdown()


@pytest.fixture(scope="function")
def instrument_with_content(tracer_provider, logger_provider):
    os.environ.update({TRACELOOP_TRACE_CONTENT: "True"})

    instrumentor = GoogleGenerativeAiInstrumentor(use_legacy_attributes=False)
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
    )

    yield instrumentor

    os.environ.pop(TRACELOOP_TRACE_CONTENT, None)
    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_no_content(tracer_provider, logger_provider):
    os.environ.update({TRACELOOP_TRACE_CONTENT: "False"})

    instrumentor = GoogleGenerativeAiInstrumentor(use_legacy_attributes=False)
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
    )

    yield instrumentor

    os.environ.pop(TRACELOOP_TRACE_CONTENT, None)
    instrumentor.uninstrument()


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization", "x-goog-api-key"]}


@pytest.fixture(autouse=True)
def environment():
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = "test_api_key"
