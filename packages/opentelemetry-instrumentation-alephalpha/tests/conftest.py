"""Unit tests configuration module."""

import os

import pytest
from aleph_alpha_client import Client
from opentelemetry.instrumentation.alephalpha import (
    TRACELOOP_TRACE_CONTENT,
    AlephAlphaInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)
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
def aleph_alpha_client():
    return Client(token=os.environ.get("AA_TOKEN"), host="https://api.aleph-alpha.com")


@pytest.fixture(scope="function")
def instrument_legacy(tracer_provider):
    instrumentor = AlephAlphaInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
    )

    yield instrumentor

    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_content(tracer_provider, logger_provider):
    os.environ.update({TRACELOOP_TRACE_CONTENT: "True"})

    instrumentor = AlephAlphaInstrumentor(use_legacy_attributes=False)
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

    instrumentor = AlephAlphaInstrumentor(use_legacy_attributes=False)
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
    )

    yield instrumentor

    os.environ.pop(TRACELOOP_TRACE_CONTENT, None)
    instrumentor.uninstrument()


@pytest.fixture(autouse=True)
def environment():
    if "AA_TOKEN" not in os.environ:
        os.environ["AA_TOKEN"] = "test_api_key"


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization"], "decode_compressed_response": True}
