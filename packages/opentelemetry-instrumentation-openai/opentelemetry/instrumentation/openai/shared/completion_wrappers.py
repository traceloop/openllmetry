import logging

from opentelemetry import context as context_api

from opentelemetry.semconv.ai import SpanAttributes, LLMRequestTypeValues

from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.instrumentation.openai.utils import _with_tracer_wrapper, start_as_current_span_async
from opentelemetry.instrumentation.openai.shared import (
    _set_request_attributes,
    _set_span_attribute,
    _set_functions_attributes,
    _set_response_attributes,
    _build_from_streaming_response,
    is_streaming_response,
    should_send_prompts,
)

from opentelemetry.instrumentation.openai.utils import is_openai_v1

from opentelemetry.trace import get_tracer, SpanKind
from opentelemetry.trace.status import Status, StatusCode

SPAN_NAME = "openai.completion"
LLM_REQUEST_TYPE = LLMRequestTypeValues.COMPLETION

logger = logging.getLogger(__name__)


@_with_tracer_wrapper
def completion_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    with tracer.start_as_current_span(SPAN_NAME, kind=SpanKind.CLIENT) as span:
        _set_request_attributes(span, LLM_REQUEST_TYPE, kwargs)
        if should_send_prompts():
            _set_prompts(span, kwargs.get("prompt"))
            _set_functions_attributes(span, kwargs.get("functions"))

        response = wrapped(*args, **kwargs)

        if is_streaming_response(response):
            # TODO: WTH is this?
            _build_from_streaming_response(span, LLM_REQUEST_TYPE, response)
        else:
            _set_response_attributes(span, response.__dict__ if is_openai_v1() else response)

        if should_send_prompts():
            _set_completions(span, kwargs.get("messages"))

        return response


@_with_tracer_wrapper
async def acompletion_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    async with start_as_current_span_async(tracer=tracer, name=SPAN_NAME, kind=SpanKind.CLIENT) as span:
        _set_request_attributes(span, LLM_REQUEST_TYPE, kwargs)
        if should_send_prompts():
            _set_prompts(span, kwargs.get("prompt"))
            _set_functions_attributes(span, kwargs.get("functions"))

        response = await wrapped(*args, **kwargs)

        if is_streaming_response(response):
            # TODO: WTH is this?
            _build_from_streaming_response(span, LLM_REQUEST_TYPE, response)
        else:
            _set_response_attributes(span, response.__dict__ if is_openai_v1() else response)

        if should_send_prompts():
            _set_completions(span, kwargs.get("messages"))
    
        return response


def _set_prompts(span, prompt):
    if not span.is_recording() or not prompt:
        return

    try:
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_PROMPTS}.0.user",
            prompt[0] if isinstance(prompt, list) else prompt,
        )
    except Exception as ex:  # pylint: disable=broad-except
        logger.warning("Failed to set prompts for openai span, error: %s", str(ex))


def _set_completions(span, choices):
    if not span.is_recording() or not choices:
        return

    try:
        for choice in choices:
            if is_openai_v1() and not isinstance(choice, dict):
                choice = choice.__dict__

            index = choice.get("index")
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
            _set_span_attribute(span, f"{prefix}.finish_reason", choice.get("finish_reason"))
            _set_span_attribute(span, f"{prefix}.content", choice.get("text"))
    except Exception as e:
        logger.warning("Failed to set completion attributes, error: %s", str(e))


