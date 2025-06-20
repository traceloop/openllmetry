from opentelemetry.instrumentation.replicate.utils import (
    dont_throw,
    should_send_prompts,
)
from opentelemetry.semconv_ai import SpanAttributes


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


@dont_throw
def set_input_attributes(span, args, kwargs):
    if not span.is_recording():
        return

    input_attribute = kwargs.get("input")
    if should_send_prompts():
        _set_span_attribute(
            span, f"{SpanAttributes.LLM_PROMPTS}.0.user", input_attribute.get("prompt")
        )


@dont_throw
def set_model_input_attributes(span, args, kwargs):
    if not span.is_recording():
        return

    if args is not None and len(args) > 0:
        _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, args[0])
    elif kwargs.get("version"):
        _set_span_attribute(
            span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("version").id
        )
    else:
        _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, "unknown")

    input_attribute = kwargs.get("input")

    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TEMPERATURE, input_attribute.get("temperature")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TOP_P, input_attribute.get("top_p")
    )


@dont_throw
def set_response_attributes(span, response):
    if should_send_prompts():
        if isinstance(response, list):
            for index, item in enumerate(response):
                prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
                _set_span_attribute(span, f"{prefix}.content", item)
        elif isinstance(response, str):
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", response
            )
