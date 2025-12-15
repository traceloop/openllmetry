"""Unit tests configuration module."""

import os

import pytest
from opentelemetry.instrumentation.bedrock import BedrockInstrumentor
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.instrumentation.langchain.config import Config
from opentelemetry.instrumentation.langchain.utils import TRACELOOP_TRACE_CONTENT
from opentelemetry.instrumentation.langchain.version import __version__
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
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


@pytest.fixture(scope="session", name="reader")
def fixture_reader():
    reader = InMemoryMetricReader(
        {Counter: AggregationTemporality.DELTA, Histogram: AggregationTemporality.DELTA}
    )
    return reader


@pytest.fixture(scope="session", name="meter_provider")
def fixture_meter_provider(reader):
    resource = Resource.create()
    meter_provider = MeterProvider(metric_readers=[reader], resource=resource)

    return meter_provider


@pytest.fixture(scope="session")
def instrument_legacy(reader, tracer_provider, meter_provider):
    openai_instrumentor = OpenAIInstrumentor()
    openai_instrumentor.instrument(
        tracer_provider=tracer_provider, meter_provider=meter_provider
    )

    langchain_instrumentor = LangchainInstrumentor()
    langchain_instrumentor.instrument(
        tracer_provider=tracer_provider, meter_provider=meter_provider
    )

    bedrock_instrumentor = BedrockInstrumentor()
    bedrock_instrumentor.instrument(tracer_provider=tracer_provider)

    yield

    openai_instrumentor.uninstrument()
    langchain_instrumentor.uninstrument()
    bedrock_instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_content(instrument_legacy, logger_provider):
    os.environ.update({TRACELOOP_TRACE_CONTENT: "True"})

    Config.use_legacy_attributes = False
    Config.event_logger = logger_provider.get_logger(
        __name__, __version__
    )
    instrumentor = instrument_legacy

    yield instrumentor

    Config.use_legacy_attributes = True
    Config.event_logger = None
    os.environ.pop(TRACELOOP_TRACE_CONTENT, None)


@pytest.fixture(scope="function")
def instrument_with_no_content(instrument_legacy, logger_provider):
    os.environ.update({TRACELOOP_TRACE_CONTENT: "False"})

    Config.use_legacy_attributes = False
    Config.event_logger = logger_provider.get_logger(
        __name__, __version__
    )
    instrumentor = instrument_legacy

    yield instrumentor

    Config.use_legacy_attributes = True
    Config.event_logger = None
    os.environ.pop(TRACELOOP_TRACE_CONTENT, None)


@pytest.fixture(autouse=True)
def clear_exporter(span_exporter):
    span_exporter.clear()


@pytest.fixture(autouse=True)
def environment():
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "test"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = "test"
    if not os.environ.get("COHERE_API_KEY"):
        os.environ["COHERE_API_KEY"] = "test"
    if not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = "test"


@pytest.fixture(scope="module")
def vcr_config():
    def before_record_request(request):
        if hasattr(request, "body") and request.body:
            import json

            try:
                if isinstance(request.body, (str, bytes)):
                    body_str = (
                        request.body.decode("utf-8")
                        if isinstance(request.body, bytes)
                        else request.body
                    )
                    body_data = json.loads(body_str)
                    if "api_key" in body_data:
                        body_data["api_key"] = "FILTERED"
                        request.body = json.dumps(body_data)
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                pass
        return request

    return {
        "filter_headers": ["authorization", "x-api-key"],
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
        "before_record_request": before_record_request,
    }
