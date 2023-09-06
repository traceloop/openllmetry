import contextvars
import logging
import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor, Tracer
import importlib.util

from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.trace import get_tracer_provider, ProxyTracerProvider
from traceloop.sdk.semconv import SpanAttributes
from traceloop.sdk.tracing import NoLogSpanBatchProcessor

TRACER_NAME = "traceloop.tracer"
TRACELOOP_API_ENDPOINT = "https://api.traceloop.dev/v1/traces"
EXCLUDED_URLS = "api.openai.com,openai.azure.com"

ctx_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id"
)
ctx_workflow_name: contextvars.ContextVar[str] = contextvars.ContextVar("workflow_name")


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


def init_instrumentations():
    init_openai_instrumentor()
    init_requests_instrumentor()
    init_urllib3_instrumentor()
    init_pymysql_instrumentor()


def span_processor_on_start(span, parent_context):
    workflow_name = ctx_workflow_name.get()
    if workflow_name is not None:
        span.set_attribute(SpanAttributes.TRACELOOP_WORKFLOW_NAME, workflow_name)

    correlation_id = ctx_correlation_id.get()
    if correlation_id is not None:
        span.set_attribute(SpanAttributes.TRACELOOP_CORRELATION_ID, correlation_id)


class Tracing:
    __initialized: bool = False
    __spans_exporter: SpanExporter = None
    __spans_processor: SpanProcessor = None
    __otel_tracer: Tracer = None
    __otel_tracer_provider: TracerProvider = None

    @staticmethod
    def init(app_name: Optional[str] = None):
        if Tracing.__initialized:
            return

        if app_name is not None:
            print(f"Traceloop tracing initialized with app name: {app_name}")
            os.environ["OTEL_SERVICE_NAME"] = app_name

        Tracing.init_spans_exporter()
        Tracing.init_tracer_provider()
        Tracing.init_spans_processor()
        Tracing.__otel_tracer_provider.add_span_processor(Tracing.__spans_processor)

        init_instrumentations()

        Tracing.__initialized = True

    @staticmethod
    def get_tracer():
        if not Tracing.__initialized:
            raise Exception("Traceloop tracing is not initialized")

        return Tracing.__otel_tracer_provider.get_tracer(TRACER_NAME)

    @staticmethod
    def init_spans_exporter():
        api_key = os.getenv("TRACELOOP_API_KEY")
        api_endpoint = os.getenv("TRACELOOP_API_ENDPOINT") or TRACELOOP_API_ENDPOINT

        Tracing.__spans_exporter = OTLPSpanExporter(
            endpoint=api_endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
            },
        )

    @staticmethod
    def init_tracer_provider():
        tracer_provider: TracerProvider
        default_provider: TracerProvider = get_tracer_provider()

        if isinstance(default_provider, ProxyTracerProvider):
            tracer_provider = TracerProvider()
            trace.set_tracer_provider(tracer_provider)
        elif not hasattr(default_provider, "add_span_processor"):
            logging.error(
                "Cannot add span processor to the default provider since it doesn't support it"
            )
            return
        else:
            tracer_provider = default_provider

        Tracing.__otel_tracer_provider = tracer_provider

    @staticmethod
    def init_spans_processor():
        Tracing.__spans_processor = NoLogSpanBatchProcessor(Tracing.__spans_exporter)
        Tracing.__spans_processor.on_start = span_processor_on_start

    @staticmethod
    def flush():
        Tracing.__spans_processor.force_flush()

    @staticmethod
    def set_correlation_id(correlation_id: str):
        ctx_correlation_id.set(correlation_id)

    @staticmethod
    def set_workflow_name(workflow_name: str):
        ctx_workflow_name.set(workflow_name)
