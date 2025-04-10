from typing import Collection
import mcp
import json

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper as _W
from opentelemetry.trace import Tracer
from opentelemetry.trace.status import Status, StatusCode
from langtrace_python_sdk.utils.llm import set_span_attributes
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.trace.propagation import set_span_in_context

from opentelemetry.instrumentation.mcp.version import __version__

_instruments = ("mcp >= 1.3.0",)

class McpInstrumentor(BaseInstrumentor):
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments
    
    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        try:
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
        except Exception as e:
            print(f"Error : {e}")

    def _uninstrument(self, **kwargs):
        pass


def serialize(request):
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
        for attrib in request.__dict__:
            print(attrib, request.__dict__[attrib])
            if type(request.__dict__[attrib]) in [bool, str, int, float, type(None)]:
                result[str(attrib)] = request.__dict__[attrib]
            else:
                result[str(attrib)] = serialize(request.__dict__[attrib])
        return json.dumps(result)

def patch_mcp_server(operation_name, tracer: Tracer):
    def traced_method(wrapped, instance, args, kwargs):
        print(f"Server : \nArgs: {args}\nKwargs : {kwargs}\n")
        attributes = {"name":f"{operation_name}", "method":"sse"}
        attributes['args'] = serialize(args[1])

        method = args[1].method
        carrier = {'traceparent': args[1].params.meta.traceparent}
        ctx = TraceContextTextMapPropagator().extract(carrier=carrier)
        with tracer.start_as_current_span(f"{method}", context=ctx) as span:
            set_span_attributes(span, attributes)
            try:
                result = wrapped(*args, **kwargs)
                if result:
                    span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                print(f"Exception {e}")
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    return traced_method

def patch_mcp_client(operation_name, tracer: Tracer):
    def traced_method(wrapped, instance, args, kwargs):
        print(f"Client : \nArgs: {args}\nKwargs : {kwargs}\n")
        attributes = {"name":f"{operation_name}", "method":"sse"}
        attributes['args'] = serialize(args[0].root)

        meta = {}
        method = args[0].root.method
        params = args[0].root.params
        if params is None:
            args[0].root.params = mcp.types.RequestParams()
            meta = {}
        else:
            meta = args[0].root.params.meta
            if meta is None:
                meta = {}

        with tracer.start_as_current_span(f"{method}") as span:
            set_span_attributes(span, attributes)
            ctx = set_span_in_context(span)
            TraceContextTextMapPropagator().inject(meta, ctx)
            args[0].root.params.meta = meta
            try:
                result = wrapped(*args, **kwargs)
                if result:
                    span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                print(f"Exception {e}")
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    return traced_method
