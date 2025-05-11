import json

from opentelemetry.semconv_ai import (
    SpanAttributes,
)


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def set_stream_response_attributes(span, response_body):
    if not span.is_recording():
        return

    _set_span_attribute(
        span, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json.dumps(response_body)
    )


def set_call_span_attributes(span, kwargs, response):
    if not span.is_recording():
        return

    endpoint_name = kwargs.get("EndpointName")
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, endpoint_name)


def set_call_request_attributes(span, kwargs):
    if not span.is_recording():
        return

    request_body = json.loads(kwargs.get("Body"))
    _set_span_attribute(
        span, SpanAttributes.TRACELOOP_ENTITY_INPUT, json.dumps(request_body)
    )


def set_call_response_attributes(span, response):
    if not span.is_recording():
        return
    response_body = json.loads(response.get("Body").read())

    _set_span_attribute(
        span, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json.dumps(response_body)
    )
