from opentelemetry.semconv_ai import SpanAttributes

from opentelemetry.instrumentation.writer.utils import (
    dont_throw,
    set_span_attribute,
    should_send_prompts,
)


@dont_throw
def set_input_attributes(span, kwargs):
    if not span.is_recording():
        return

    if should_send_prompts():
        if kwargs.get("prompt") is not None:
            set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
            set_span_attribute(
                span, f"{SpanAttributes.LLM_PROMPTS}.0.content", kwargs.get("prompt")
            )

        elif kwargs.get("messages") is not None:
            for i, message in enumerate(kwargs.get("messages")):
                set_span_attribute(
                    span, f"{SpanAttributes.LLM_PROMPTS}.{i}.role", message.get("role")
                )
                set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                    message.get("content"),
                )
            # TODO add tool calls setter


@dont_throw
def set_model_input_attributes(span, kwargs):
    if not span.is_recording():
        return

    set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
    set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_tokens")
    )
    set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TEMPERATURE, kwargs.get("temperature")
    )
    set_span_attribute(span, SpanAttributes.LLM_REQUEST_TOP_P, kwargs.get("top_p"))
    set_span_attribute(span, SpanAttributes.LLM_CHAT_STOP_SEQUENCES, kwargs.get("stop"))
    set_span_attribute(
        span, SpanAttributes.LLM_IS_STREAMING, kwargs.get("stream") or False
    )
