"""Unit tests configuration module."""

import asyncio
import os
from importlib.metadata import version

import pytest
from opentelemetry.instrumentation.together import TogetherAiInstrumentor
from opentelemetry.instrumentation.together.utils import (
    TRACELOOP_TRACE_CONTENT,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from together import AsyncTogether, Together

# The suite runs once per together major line (see the `test` target in
# project.json). 1.x and 2.x/3.x call different hosts, so each line replays its
# own cassettes. 2.x shares the 3.x SDK and is covered by the 3.x run.
TOGETHER_MAJOR = int(version("together").split(".")[0])
SDK_LINE = "v1" if TOGETHER_MAJOR < 2 else "v3"

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


@pytest.fixture(scope="function", name="reader")
def fixture_reader():
    return InMemoryMetricReader()


@pytest.fixture(scope="function", name="meter_provider")
def fixture_meter_provider(reader):
    return MeterProvider(metric_readers=[reader])


class _Caller:
    """Calls `<resource>.create` on a sync or async together client."""

    def __init__(self, is_async):
        self.is_async = is_async
        client_cls = AsyncTogether if is_async else Together
        self.client = client_cls(api_key=os.environ.get("TOGETHER_API_KEY"))

    def chat(self, **kwargs):
        return self._run(self.client.chat.completions.create(**kwargs))

    def completion(self, **kwargs):
        return self._run(self.client.completions.create(**kwargs))

    def _run(self, result):
        return asyncio.run(result) if self.is_async else result

    def request_patch_target(self):
        """The SDK's internal request method, to inject transport errors."""
        if TOGETHER_MAJOR < 2:
            method = "arequest" if self.is_async else "request"
            return f"together.abstract.api_requestor.APIRequestor.{method}"
        client = "AsyncAPIClient" if self.is_async else "SyncAPIClient"
        return f"together._base_client.{client}.request"


@pytest.fixture(params=["sync", "async"])
def caller(request):
    return _Caller(is_async=request.param == "async")


@pytest.fixture(scope="function")
def instrument_legacy(tracer_provider, meter_provider):
    instrumentor = TogetherAiInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor

    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_content(tracer_provider, logger_provider, meter_provider):
    os.environ.update({TRACELOOP_TRACE_CONTENT: "True"})

    instrumentor = TogetherAiInstrumentor(use_attributes=False)
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor

    os.environ.pop(TRACELOOP_TRACE_CONTENT, None)
    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_no_content(tracer_provider, logger_provider, meter_provider):
    os.environ.update({TRACELOOP_TRACE_CONTENT: "False"})

    instrumentor = TogetherAiInstrumentor(use_attributes=False)
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
    if "TOGETHER_API_KEY" not in os.environ:
        os.environ["TOGETHER_API_KEY"] = "test_api_key"


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    test_dir = os.path.dirname(request.module.__file__)
    module_name = request.module.__name__.rsplit(".", 1)[-1]
    return os.path.join(test_dir, "cassettes", SDK_LINE, module_name)


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization"], "decode_compressed_response": True}
