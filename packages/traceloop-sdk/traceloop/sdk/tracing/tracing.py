import logging
import os
import importlib.util
import requests

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter, BatchSpanProcessor
from opentelemetry.trace import get_tracer_provider, ProxyTracerProvider
from opentelemetry.context import get_value, attach, set_value
from opentelemetry.util.re import parse_env_headers
from traceloop.sdk.semconv import SpanAttributes

TRACER_NAME = "traceloop.tracer"
TRACELOOP_API_ENDPOINT = "https://api.traceloop.dev"
EXCLUDED_URLS = "api.openai.com,openai.azure.com"


class TracerWrapper(object):
    initialized: bool = False

    def __new__(cls) -> "TracerWrapper":
        if not hasattr(cls, "instance"):
            cls.instance = super(TracerWrapper, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        self.__spans_exporter: SpanExporter = init_spans_exporter()
        self.__tracer_provider: TracerProvider = init_tracer_provider()
        self.__spans_processor: SpanProcessor = BatchSpanProcessor(
            self.__spans_exporter
        )
        self.__spans_processor.on_start = span_processor_on_start
        self.__tracer_provider.add_span_processor(self.__spans_processor)

        init_instrumentations()

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


def init_spans_exporter() -> SpanExporter:
    api_key = os.getenv("TRACELOOP_API_KEY")
    api_endpoint = os.getenv("TRACELOOP_API_ENDPOINT") or TRACELOOP_API_ENDPOINT
    headers = os.getenv("TRACELOOP_HEADERS") or {}

    if isinstance(headers, str):
        headers = parse_env_headers(headers)
        
    return OTLPSpanExporter(
        endpoint=f"{api_endpoint}/v1/traces",
        headers={
            "Authorization": f"Bearer {api_key}",
        } | headers,
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
    init_openai_instrumentor()
    init_requests_instrumentor()
    init_urllib3_instrumentor()
    init_pymysql_instrumentor()


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
