import logging

from opentelemetry import context as context_api
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY
)
from opentelemetry.semconv.ai import SpanAttributes, LLMRequestTypeValues
from opentelemetry.instrumentation.haystack.utils import (
    with_tracer_wrapper,
    set_span_attribute,
)

logger = logging.getLogger(__name__)


def _set_input_attributes(span, llm_request_type, kwargs):
    base_payload = kwargs.get("base_payload")
    set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MODEL, base_payload.get("model")
    )
    set_span_attribute(
        span, SpanAttributes.LLM_TEMPERATURE, base_payload.get("temperature")
    )
    set_span_attribute(span, SpanAttributes.LLM_TOP_P, base_payload.get("top_p"))
    set_span_attribute(
        span,
        SpanAttributes.LLM_FREQUENCY_PENALTY,
        base_payload.get("frequency_penalty"),
    )
    set_span_attribute(
        span, SpanAttributes.LLM_PRESENCE_PENALTY, base_payload.get("presence_penalty")
    )

    set_span_attribute(
        span, f"{SpanAttributes.LLM_PROMPTS}.0.user", kwargs.get("prompt")
    )

    return


def _set_span_completions(span, llm_request_type, choices):
    if choices is None:
        return

    for index, message in enumerate(choices):
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"

        if llm_request_type == LLMRequestTypeValues.CHAT:
            if message is not None:
                set_span_attribute(span, f"{prefix}.role", "assistant")
                set_span_attribute(span, f"{prefix}.content", message)
        elif llm_request_type == LLMRequestTypeValues.COMPLETION:
            set_span_attribute(span, f"{prefix}.content", message)


def _set_response_attributes(span, llm_request_type, response):
    _set_span_completions(span, llm_request_type, response)

    return


def _llm_request_type_by_object(object_name):
    if object_name == "OpenAIInvocationLayer":
        return LLMRequestTypeValues.COMPLETION
    elif object_name == "ChatGPTInvocationLayer":
        return LLMRequestTypeValues.CHAT
    else:
        return LLMRequestTypeValues.UNKNOWN


@with_tracer_wrapper
def wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    llm_request_type = _llm_request_type_by_object(to_wrap.get("object"))
    with tracer.start_as_current_span(
        "openai.chat"
        if llm_request_type == LLMRequestTypeValues.CHAT
        else "openai.completion",
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_VENDOR: "OpenAI",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    ) as span:
        try:
            if span.is_recording():
                _set_input_attributes(span, llm_request_type, kwargs)

        except Exception as ex:  # pylint: disable=broad-except
            logger.warning(
                "Failed to set input attributes for openai span, error: %s", str(ex)
            )

        response = wrapped(*args, **kwargs)

        if response:
            try:
                if span.is_recording():
                    _set_response_attributes(span, llm_request_type, response)

            except Exception as ex:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to set response attributes for openai span, error: %s",
                    str(ex),
                )
            if span.is_recording():
                span.set_status(Status(StatusCode.OK))

        return response
