from opentelemetry.instrumentation.transformers.utils import (
    dont_throw,
    should_send_prompts,
)
from opentelemetry.semconv_ai import (
    SpanAttributes,
)


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


@dont_throw
def set_input_attributes(span, instance, args, kwargs):
    if not span.is_recording():
        return

    if args and len(args) > 0:
        prompts_list = args[0]
    else:
        prompts_list = kwargs.get("args")

    _set_span_prompts(span, prompts_list)


@dont_throw
def set_model_input_attributes(span, instance):
    if not span.is_recording():
        return

    forward_params = instance._forward_params

    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MODEL, instance.model.config.name_or_path
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_SYSTEM, instance.model.config.model_type
    )
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TYPE, "completion")
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TEMPERATURE, forward_params.get("temperature")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TOP_P, forward_params.get("top_p")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, forward_params.get("max_length")
    )
    _set_span_attribute(
        span,
        SpanAttributes.LLM_REQUEST_REPETITION_PENALTY,
        forward_params.get("repetition_penalty"),
    )


@dont_throw
def set_response_attributes(span, response):
    if response and span.is_recording():
        if len(response) > 0:
            _set_span_completions(span, response)


def _set_span_completions(span, completions):
    if completions is None or not should_send_prompts():
        return

    for i, completion in enumerate(completions):
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{i}"
        _set_span_attribute(span, f"{prefix}.content", completion.get("generated_text"))


def _set_span_prompts(span, messages):
    if messages is None or not should_send_prompts():
        return

    if isinstance(messages, str):
        messages = [messages]

    for i, msg in enumerate(messages):
        prefix = f"{SpanAttributes.LLM_PROMPTS}.{i}"
        _set_span_attribute(span, f"{prefix}.content", msg)
