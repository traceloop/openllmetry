import httpx
from opentelemetry.sdk.trace import Span
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.trace.propagation import get_current_span
from unittest.mock import MagicMock


# from: https://stackoverflow.com/a/41599695/2749989
def spy_decorator(method_to_decorate):
    mock = MagicMock()

    def wrapper(self, *args, **kwargs):
        mock(*args, **kwargs)
        return method_to_decorate(self, *args, **kwargs)

    wrapper.mock = mock
    return wrapper


def assert_request_contains_tracecontext(request: httpx.Request, expected_span: Span):
    assert TraceContextTextMapPropagator._TRACEPARENT_HEADER_NAME in request.headers
    ctx = TraceContextTextMapPropagator().extract(request.headers)
    request_span_context = get_current_span(ctx).get_span_context()
    expected_span_context = expected_span.get_span_context()

    assert request_span_context.trace_id == expected_span_context.trace_id
    assert request_span_context.span_id == expected_span_context.span_id
