"""Unit tests configuration module."""

import os

import pytest
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from opentelemetry._logs import get_logger
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.openai.shared.config import Config
from opentelemetry.instrumentation.openai.utils import TRACELOOP_TRACE_CONTENT
from opentelemetry.instrumentation.openai.version import __version__
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


@pytest.fixture(autouse=True)
def environment():
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "test_api_key"
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        os.environ["AZURE_OPENAI_API_KEY"] = "test_azure_api_key"
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://traceloop-stg.openai.azure.com/"


@pytest.fixture
def openai_client():
    return OpenAI()


@pytest.fixture
def mock_openai_client():
    return OpenAI(
        api_key="test-key",
        base_url="http://localhost:5002/v1/"
    )


@pytest.fixture
def deepseek_client():
    return OpenAI(
        api_key="test-key",
        base_url="https://api.deepseek.com/v1"
    )


@pytest.fixture
def vllm_openai_client():
    return OpenAI(base_url="http://localhost:8000/v1")


@pytest.fixture
def azure_openai_client():
    return AzureOpenAI(
        api_version="2024-02-01",
    )


@pytest.fixture
def async_azure_openai_client():
    return AsyncAzureOpenAI(
        api_version="2024-02-01",
    )


@pytest.fixture
def async_openai_client():
    return AsyncOpenAI()


@pytest.fixture
def async_vllm_openai_client():
    return AsyncOpenAI(base_url="http://localhost:8000/v1")


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
    async def upload_base64_image(*args):
        return "/some/url"

    instrumentor = OpenAIInstrumentor(
        enrich_assistant=True,
        upload_base64_image=upload_base64_image,
    )
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor

    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_content(
    instrument_legacy, reader, tracer_provider, logger_provider, meter_provider
):
    os.environ.update({TRACELOOP_TRACE_CONTENT: "True"})

    instrumentor = instrument_legacy
    Config.use_legacy_attributes = False
    Config.event_logger = get_logger(
        __name__, __version__, logger_provider=logger_provider
    )

    yield instrumentor

    Config.use_legacy_attributes = True
    Config.event_logger = None
    os.environ.pop(TRACELOOP_TRACE_CONTENT, None)
    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_no_content(
    instrument_legacy, reader, tracer_provider, logger_provider, meter_provider
):
    os.environ.update({TRACELOOP_TRACE_CONTENT: "False"})

    instrumentor = instrument_legacy
    Config.use_legacy_attributes = False
    Config.event_logger = get_logger(
        __name__, __version__, logger_provider=logger_provider
    )

    yield instrumentor

    Config.use_legacy_attributes = True
    Config.event_logger = None
    os.environ.pop(TRACELOOP_TRACE_CONTENT, None)
    instrumentor.uninstrument()


@pytest.fixture(autouse=True)
def clear_exporter(span_exporter):
    span_exporter.clear()


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization", "api-key"]}
