"""Unit tests configuration module."""

import os

import pytest
import transformers
from opentelemetry.instrumentation.transformers import TransformersInstrumentor
from opentelemetry.instrumentation.transformers.utils import TRACELOOP_TRACE_CONTENT
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
def transformers_pipeline():
    return transformers.pipeline(task="text-generation", model="gpt2")


@pytest.fixture(scope="function")
def instrument_legacy(tracer_provider):
    instrumentor = TransformersInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
    )

    yield instrumentor

    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_content(tracer_provider, logger_provider):
    os.environ.update({TRACELOOP_TRACE_CONTENT: "True"})

    instrumentor = TransformersInstrumentor(use_legacy_attributes=False)
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

    instrumentor = TransformersInstrumentor(use_legacy_attributes=False)
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
    )

    yield instrumentor

    os.environ.pop(TRACELOOP_TRACE_CONTENT, None)
    instrumentor.uninstrument()
