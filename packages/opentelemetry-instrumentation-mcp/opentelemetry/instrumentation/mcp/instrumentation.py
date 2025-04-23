from typing import Collection
import mcp
import json

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper as _W
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.trace.propagation import set_span_in_context
from opentelemetry.semconv_ai import SpanAttributes

from opentelemetry.instrumentation.mcp.version import __version__

_instruments = ("mcp >= 1.3.0",)

class McpInstrumentor(BaseInstrumentor):
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        _W(
            "mcp.server.lowlevel.server",
            "Server._handle_request",
            patch_mcp_server("Server._handle_request", tracer),
        )
        _W(
            "mcp.shared.session",
            "BaseSession.send_request",
            patch_mcp_client("BaseSession.send_request", tracer),
        )

    def _uninstrument(self, **kwargs):
        pass


def with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(operation_name, tracer):
        def wrapper(wrapped, instance, args, kwargs):
            return func(operation_name, tracer, wrapped, instance, args, kwargs)
        return wrapper
    return _with_tracer

def serialize(request, depth=0, max_depth=2):
    if depth > max_depth:
        return {}
    depth += 1

    def is_serializable(request):
        try:
            json.dumps(request)
            return True
        except Exception:
            return False
        
    if is_serializable(request):
        return json.dumps(request)
    else:
        result = {}
        try:
            if hasattr(request, '__dict__'):
                for attrib in request.__dict__:
                    if type(request.__dict__[attrib]) in [bool, str, int, float, type(None)]:
                        result[str(attrib)] = request.__dict__[attrib]
                    else:
                        result[str(attrib)] = serialize(request.__dict__[attrib], depth)
        except Exception as e:
            pass
        return json.dumps(result)

@with_tracer_wrapper
def patch_mcp_server(operation_name, tracer, wrapped, instance, args, kwargs):
    method = args[1].method
    carrier = None
    ctx = None
    if hasattr(args[1], 'params'):
        if hasattr(args[1].params, 'meta'):
            if hasattr(args[1].params.meta, 'traceparent'):
                carrier = {'traceparent': args[1].params.meta.traceparent}
    if carrier:
        ctx = TraceContextTextMapPropagator().extract(carrier=carrier)
    with tracer.start_as_current_span(f"{method}", context=ctx) as span:
        span.set_attribute(SpanAttributes.MCP_METHOD_NAME, f"{method}")
        if hasattr(args[1], 'id'):
            span.set_attribute(SpanAttributes.MCP_REQUEST_ID, f"{args[1].id}")
        if hasattr(args[2], '_init_options'):
            span.set_attribute(SpanAttributes.MCP_SESSION_INIT_OPTIONS, f"{args[2]._init_options}")
        span.set_attribute(SpanAttributes.MCP_REQUEST_ARGUMENT,f"{serialize(args[1])}")
        try:
            result = wrapped(*args, **kwargs)
            if result:
                span.set_status(Status(StatusCode.OK))
            return result
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

@with_tracer_wrapper
def patch_mcp_client(operation_name, tracer, wrapped, instance, args, kwargs):
    meta = None
    method = None
    params = None
    if hasattr(args[0].root, 'method'):
        method = args[0].root.method
    if hasattr(args[0].root, 'params'):
        params = args[0].root.params
    if params is None:
        args[0].root.params = mcp.types.RequestParams()
        meta = {}
    else:
        if hasattr(args[0].root.params, 'meta'):
            meta = args[0].root.params.meta
        if meta is None:
            meta = {}

    with tracer.start_as_current_span(f"{method}") as span:
        span.set_attribute(SpanAttributes.MCP_METHOD_NAME, f"{method}")
        span.set_attribute(SpanAttributes.MCP_REQUEST_ARGUMENT,f"{serialize(args[0])}")
        ctx = set_span_in_context(span)
        TraceContextTextMapPropagator().inject(meta, ctx)
        args[0].root.params.meta = meta
        try:
            result = wrapped(*args, **kwargs)
            if result:
                span.set_status(Status(StatusCode.OK))
            return result
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
