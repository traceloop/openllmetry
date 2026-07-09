"""Unit tests configuration module."""

import os

import pytest
from opentelemetry.instrumentation.chromadb import ChromaInstrumentor
from opentelemetry.instrumentation.cohere import CohereInstrumentor
from opentelemetry.instrumentation.llamaindex import LlamaIndexInstrumentor
from opentelemetry.instrumentation.llamaindex.config import Config
from opentelemetry.instrumentation.llamaindex.utils import TRACELOOP_TRACE_CONTENT
from opentelemetry.instrumentation.llamaindex.version import __version__
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytest_plugins = []


@pytest.fixture(scope="session", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(scope="session", name="tracer_provider")
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


@pytest.fixture(scope="session")
def instrument_legacy(tracer_provider):
    openai_instrumentor = OpenAIInstrumentor()
    chroma_instrumentor = ChromaInstrumentor()
    cohere_instrumentor = CohereInstrumentor()
    instrumentor = LlamaIndexInstrumentor()

    openai_instrumentor.instrument(tracer_provider=tracer_provider)
    chroma_instrumentor.instrument(tracer_provider=tracer_provider)
    cohere_instrumentor.instrument(tracer_provider=tracer_provider)
    instrumentor.instrument(tracer_provider=tracer_provider)

    yield instrumentor

    openai_instrumentor.uninstrument()
    chroma_instrumentor.uninstrument()
    cohere_instrumentor.uninstrument()
    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_content(instrument_legacy, logger_provider):
    os.environ.update({TRACELOOP_TRACE_CONTENT: "True"})

    instrumentor = instrument_legacy
    Config.use_legacy_attributes = False
    Config.event_logger = logger_provider.get_logger(
        __name__, __version__
    )

    yield instrumentor

    Config.use_legacy_attributes = True
    Config.event_logger = None
    os.environ.pop(TRACELOOP_TRACE_CONTENT, None)


@pytest.fixture(scope="function")
def instrument_with_no_content(instrument_legacy, logger_provider):
    os.environ.update({TRACELOOP_TRACE_CONTENT: "False"})

    instrumentor = instrument_legacy
    Config.use_legacy_attributes = False
    Config.event_logger = logger_provider.get_logger(
        __name__, __version__
    )

    yield instrumentor

    Config.use_legacy_attributes = True
    Config.event_logger = None
    os.environ.pop(TRACELOOP_TRACE_CONTENT, None)


@pytest.fixture(autouse=True)
def clear_exporter(span_exporter):
    span_exporter.clear()


@pytest.fixture(autouse=True)
def environment():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "test_api_key"
    if "COHERE_API_KEY" not in os.environ:
        os.environ["COHERE_API_KEY"] = "test_api_key"
    if "LLAMA_CLOUD_API_KEY" not in os.environ:
        os.environ["LLAMA_CLOUD_API_KEY"] = "test_api_key"


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization", "api-key"],
        "ignore_hosts": ["raw.githubusercontent.com"],
    }


def pytest_collection_modifyitems(items):
    move_last = []
    tests = []
    for item in items:
        # These tests are modifying imports and monkey patch python runtime
        # it could lead to instability and multiple instrumentation of the code
        # we move it as last tests to run to avoid side-effects.
        if item.name.startswith("test_instrumentation"):
            move_last.append(item)
        else:
            tests.append(item)
    items[:] = tests + move_last
