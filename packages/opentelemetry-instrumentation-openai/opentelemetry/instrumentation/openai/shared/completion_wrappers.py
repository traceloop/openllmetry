import logging

from opentelemetry import context as context_api

from opentelemetry.semconv.ai import SpanAttributes, LLMRequestTypeValues

from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.instrumentation.openai.utils import _with_tracer_wrapper
from opentelemetry.instrumentation.openai.shared import (
    _set_client_attributes,
    _set_request_attributes,
    _set_span_attribute,
    _set_functions_attributes,
    _set_response_attributes,
    is_streaming_response,
    should_send_prompts,
    model_as_dict,
)

from opentelemetry.instrumentation.openai.utils import is_openai_v1

from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import Status, StatusCode

SPAN_NAME = "openai.completion"
LLM_REQUEST_TYPE = LLMRequestTypeValues.COMPLETION

logger = logging.getLogger(__name__)


@_with_tracer_wrapper
def completion_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    # span needs to be opened and closed manually because the response is a generator
    span = tracer.start_span(
        SPAN_NAME,
        kind=SpanKind.CLIENT,
        attributes={SpanAttributes.LLM_REQUEST_TYPE: LLM_REQUEST_TYPE.value},
    )

    _handle_request(span, kwargs, instance)
    response = wrapped(*args, **kwargs)

    if is_streaming_response(response):
        # span will be closed after the generator is done
        return _build_from_streaming_response(span, response)
    else:
        _handle_response(response, span)

    span.end()
    return response


@_with_tracer_wrapper
async def acompletion_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    span = tracer.start_span(
        name=SPAN_NAME,
        kind=SpanKind.CLIENT,
        attributes={SpanAttributes.LLM_REQUEST_TYPE: LLM_REQUEST_TYPE.value},
    )

    _handle_request(span, kwargs, instance)
    response = await wrapped(*args, **kwargs)

    if is_streaming_response(response):
        # span will be closed after the generator is done
        return _abuild_from_streaming_response(span, response)
    else:
        _handle_response(response, span)

    span.end()
    return response


def _handle_request(span, kwargs, instance):
    _set_request_attributes(span, kwargs)
    if should_send_prompts():
        _set_prompts(span, kwargs.get("prompt"))
        _set_functions_attributes(span, kwargs.get("functions"))
    _set_client_attributes(span, instance)


def _handle_response(response, span):
    if is_openai_v1():
        response_dict = model_as_dict(response)
    else:
        response_dict = response

    _set_response_attributes(span, response_dict)

    if should_send_prompts():
        _set_completions(span, response_dict.get("choices"))


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
            index = choice.get("index")
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
            _set_span_attribute(
                span, f"{prefix}.finish_reason", choice.get("finish_reason")
            )
            _set_span_attribute(span, f"{prefix}.content", choice.get("text"))
    except Exception as e:
        logger.warning("Failed to set completion attributes, error: %s", str(e))


def _build_from_streaming_response(span, response):
    complete_response = {"choices": [], "model": ""}
    for item in response:
        item_to_yield = item
        if is_openai_v1():
            item = model_as_dict(item)

        for choice in item.get("choices"):
            index = choice.get("index")
            if len(complete_response.get("choices")) <= index:
                complete_response["choices"].append({"index": index, "text": ""})
            complete_choice = complete_response.get("choices")[index]
            if choice.get("finish_reason"):
                complete_choice["finish_reason"] = choice.get("finish_reason")

            if choice.get("text"):
                complete_choice["text"] += choice.get("text")

        yield item_to_yield

    _set_response_attributes(span, complete_response)

    if should_send_prompts():
        _set_completions(span, complete_response.get("choices"))

    span.set_status(Status(StatusCode.OK))
    span.end()


async def _abuild_from_streaming_response(span, response):
    complete_response = {"choices": [], "model": ""}
    async for item in response:
        item_to_yield = item
        if is_openai_v1():
            item = model_as_dict(item)

        for choice in item.get("choices"):
            index = choice.get("index")
            if len(complete_response.get("choices")) <= index:
                complete_response["choices"].append({"index": index, "text": ""})
            complete_choice = complete_response.get("choices")[index]
            if choice.get("finish_reason"):
                complete_choice["finish_reason"] = choice.get("finish_reason")

            complete_choice["text"] += choice.get("text")

        yield item_to_yield

    _set_response_attributes(span, complete_response)

    if should_send_prompts():
        _set_completions(span, complete_response.get("choices"))

    span.set_status(Status(StatusCode.OK))
    span.end()
