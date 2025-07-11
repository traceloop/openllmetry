import json

from opentelemetry.instrumentation.sagemaker.utils import should_send_prompts
from opentelemetry.semconv_ai import (
    SpanAttributes,
)


def _try_parse_json(value):
    """Try to decode JSON if it's a string or bytes; fallback to raw value."""
    try:
        if isinstance(value, bytes):
            value = value.decode("utf-8").strip()
        if isinstance(value, str):
            return json.loads(value)
        return value
    except (json.JSONDecodeError, UnicodeDecodeError):
        return str(value)


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def set_stream_response_attributes(span, response_body):
    if not span.is_recording() or not should_send_prompts():
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
    if not span.is_recording() or not should_send_prompts():
        return

    raw_request = kwargs.get("Body")
    request_body = _try_parse_json(raw_request)
    _set_span_attribute(
        span, SpanAttributes.TRACELOOP_ENTITY_INPUT, json.dumps(request_body)
    )


def set_call_response_attributes(span, raw_response):
    if not span.is_recording() or not should_send_prompts():
        return
    response_body = _try_parse_json(raw_response)

    _set_span_attribute(
        span, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json.dumps(response_body)
    )
