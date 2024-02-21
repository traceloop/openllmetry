import logging

from opentelemetry import context as context_api

from opentelemetry.semconv.ai import SpanAttributes, LLMRequestTypeValues

from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.instrumentation.openai.utils import (
    _with_tracer_wrapper,
    start_as_current_span_async,
)
from opentelemetry.instrumentation.openai.shared import (
    _set_request_attributes,
    _set_span_attribute,
    _set_response_attributes,
    should_send_prompts,
    model_as_dict,
)

from opentelemetry.instrumentation.openai.utils import is_openai_v1

from opentelemetry.trace import SpanKind

SPAN_NAME = "openai.embeddings"
LLM_REQUEST_TYPE = LLMRequestTypeValues.EMBEDDING

logger = logging.getLogger(__name__)


@_with_tracer_wrapper
def embeddings_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    with tracer.start_as_current_span(
        name=SPAN_NAME,
        kind=SpanKind.CLIENT,
        attributes={SpanAttributes.LLM_REQUEST_TYPE: LLM_REQUEST_TYPE.value},
    ) as span:
        _handle_request(span, kwargs)
        response = wrapped(*args, **kwargs)
        _handle_response(response, span)

        return response


@_with_tracer_wrapper
async def aembeddings_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    async with start_as_current_span_async(
        tracer=tracer,
        name=SPAN_NAME,
        kind=SpanKind.CLIENT,
        attributes={SpanAttributes.LLM_REQUEST_TYPE: LLM_REQUEST_TYPE.value},
    ) as span:
        _handle_request(span, kwargs)
        response = await wrapped(*args, **kwargs)
        _handle_response(response, span)

        return response


def _handle_request(span, kwargs):
    _set_request_attributes(span, kwargs)
    if should_send_prompts():
        _set_prompts(span, kwargs.get("input"))


def _handle_response(response, span):
    if is_openai_v1():
        response_dict = model_as_dict(response)
    else:
        response_dict = response

    _set_response_attributes(span, response_dict)


def _set_prompts(span, prompt):
    if not span.is_recording() or not prompt:
        return

    try:
        if isinstance(prompt, list):
            for i, p in enumerate(prompt):
                _set_span_attribute(
                    span, f"{SpanAttributes.LLM_PROMPTS}.{i}.content", p
                )
        else:
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.0.content",
                prompt,
            )
    except Exception as ex:  # pylint: disable=broad-except
        logger.warning("Failed to set prompts for openai span, error: %s", str(ex))
