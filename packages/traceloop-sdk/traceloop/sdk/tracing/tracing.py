import atexit
import logging
import os
import importlib.util

from colorama import Fore
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SimpleSpanProcessor,
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.trace import get_tracer_provider, ProxyTracerProvider
from opentelemetry.context import get_value, attach, set_value
from opentelemetry.util.re import parse_env_headers
from opentelemetry.semconv.ai import SpanAttributes

from traceloop.sdk.utils import is_notebook

TRACER_NAME = "traceloop.tracer"
TRACELOOP_API_ENDPOINT = "https://api.traceloop.dev"
EXCLUDED_URLS = "api.openai.com,openai.azure.com"


class TracerWrapper(object):
    def __new__(cls) -> "TracerWrapper":
        if not hasattr(cls, "instance"):
            obj = cls.instance = super(TracerWrapper, cls).__new__(cls)
            obj.__spans_exporter: SpanExporter = init_spans_exporter(
                TracerWrapper.endpoint, TracerWrapper.headers
            )
            obj.__tracer_provider: TracerProvider = init_tracer_provider()
            obj.__spans_processor: SpanProcessor = (
                SimpleSpanProcessor(obj.__spans_exporter)
                if is_notebook()
                else BatchSpanProcessor(obj.__spans_exporter)
            )
            obj.__spans_processor.on_start = span_processor_on_start
            obj.__tracer_provider.add_span_processor(obj.__spans_processor)

            init_instrumentations()

            # Force flushes for debug environments (e.g. local development)
            atexit.register(obj.exit_handler)

        return cls.instance

    def exit_handler(self):
        self.flush()

    @staticmethod
    def set_endpoint(endpoint: str, headers: dict[str, str]) -> None:
        TracerWrapper.endpoint = endpoint
        TracerWrapper.headers = headers

    @classmethod
    def verify_initialized(cls) -> bool:
        if hasattr(cls, "instance"):
            return True

        if (os.getenv("TRACELOOP_SUPPRESS_WARNINGS") or "false").lower() == "true":
            return False

        print(
            Fore.RED
            + "Warning: Traceloop not initialized, make sure you call Traceloop.init()"
        )
        print(Fore.RESET)
        return False

    def flush(self):
        self.__spans_processor.force_flush()

    def get_tracer(self):
        return self.__tracer_provider.get_tracer(TRACER_NAME)


def set_correlation_id(correlation_id: str) -> None:
    attach(set_value("correlation_id", correlation_id))


def set_workflow_name(workflow_name: str) -> None:
    attach(set_value("workflow_name", workflow_name))


def span_processor_on_start(span, parent_context):
    workflow_name = get_value("workflow_name")
    if workflow_name is not None:
        span.set_attribute(SpanAttributes.TRACELOOP_WORKFLOW_NAME, workflow_name)

    correlation_id = get_value("correlation_id")
    if correlation_id is not None:
        span.set_attribute(SpanAttributes.TRACELOOP_CORRELATION_ID, correlation_id)


def init_spans_exporter(api_endpoint: str, headers: dict[str, str]) -> SpanExporter:
    return (
        ConsoleSpanExporter()
        if os.getenv("TRACELOOP_CONSOLE")
        else OTLPSpanExporter(endpoint=f"{api_endpoint}/v1/traces", headers=headers)
    )


def init_tracer_provider() -> TracerProvider:
    provider: TracerProvider = None
    default_provider: TracerProvider = get_tracer_provider()

    if isinstance(default_provider, ProxyTracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    elif not hasattr(default_provider, "add_span_processor"):
        logging.error(
            "Cannot add span processor to the default provider since it doesn't support it"
        )
        return
    else:
        provider = default_provider

    return provider


def init_instrumentations():
    init_haystack_instrumentor()
    init_openai_instrumentor()
    init_requests_instrumentor()
    init_urllib3_instrumentor()
    init_pymysql_instrumentor()


def init_haystack_instrumentor():
    if importlib.util.find_spec("haystack") is not None:
        from opentelemetry.instrumentation.haystack import HaystackInstrumentor

        instrumentor = HaystackInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()


def init_openai_instrumentor():
    if importlib.util.find_spec("openai") is not None:
        from opentelemetry.instrumentation.openai import OpenAIInstrumentor

        instrumentor = OpenAIInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()


def init_requests_instrumentor():
    if importlib.util.find_spec("requests") is not None:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor

        instrumentor = RequestsInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument(excluded_urls=EXCLUDED_URLS)


def init_urllib3_instrumentor():
    if importlib.util.find_spec("urllib3") is not None:
        from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor

        instrumentor = URLLib3Instrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument(excluded_urls=EXCLUDED_URLS)


def init_pymysql_instrumentor():
    if importlib.util.find_spec("pymysql") is not None:
        from opentelemetry.instrumentation.pymysql import PyMySQLInstrumentor

        instrumentor = PyMySQLInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
