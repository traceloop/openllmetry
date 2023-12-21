import json
import logging

from opentelemetry import context as context_api

from opentelemetry.semconv.ai import SpanAttributes, LLMRequestTypeValues

from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.instrumentation.openai.utils import (
    _with_tracer_wrapper
)
from opentelemetry.instrumentation.openai.shared import (
    _set_request_attributes,
    _set_span_attribute,
    _set_functions_attributes,
    _set_response_attributes,
    is_streaming_response,
    should_send_prompts,
    model_as_dict,
)
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.openai.utils import is_openai_v1

SPAN_NAME = "openai.chat"
LLM_REQUEST_TYPE = LLMRequestTypeValues.CHAT

logger = logging.getLogger(__name__)


@_with_tracer_wrapper
def chat_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    # span needs to be opened and closed manually because the response is a generator
    span = tracer.start_span(SPAN_NAME, kind=SpanKind.CLIENT)

    _handle_request(span, kwargs)
    response = wrapped(*args, **kwargs)

    if is_streaming_response(response):
        # span will be closed after the generator is done
        return _build_from_streaming_response(span, response)

    _handle_response(response, span)
    span.end()

    return response


@_with_tracer_wrapper
async def achat_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    span = tracer.start_span(SPAN_NAME, kind=SpanKind.CLIENT)
    _handle_request(span, kwargs)
    response = await wrapped(*args, **kwargs)

    if is_streaming_response(response):
        # span will be closed after the generator is done
        return _abuild_from_streaming_response(span, response)

    _handle_response(response, span)
    span.end()

    return response


def _handle_request(span, kwargs):
    _set_request_attributes(span, LLM_REQUEST_TYPE, kwargs)
    if should_send_prompts():
        _set_prompts(span, kwargs.get("messages"))
        _set_functions_attributes(span, kwargs.get("functions"))


def _handle_response(response, span):
    if is_openai_v1():
        response_dict = model_as_dict(response)
    else:
        response_dict = response

    _set_response_attributes(span, response_dict)

    if should_send_prompts():
        _set_completions(span, response_dict.get("choices"))

    return response


def _set_prompts(span, messages):
    if not span.is_recording() or messages is None:
        return

    try:
        for i, msg in enumerate(messages):
            prefix = f"{SpanAttributes.LLM_PROMPTS}.{i}"
            if isinstance(msg.get("content"), str):
                content = msg.get("content")
            elif isinstance(msg.get("content"), list):
                content = json.dumps(msg.get("content"))

            _set_span_attribute(span, f"{prefix}.role", msg.get("role"))
            _set_span_attribute(span, f"{prefix}.content", content)
    except Exception as ex:  # pylint: disable=broad-except
        logger.warning("Failed to set prompts for openai span, error: %s", str(ex))


def _set_completions(span, choices):
    if choices is None:
        return

    for choice in choices:
        index = choice.get("index")
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
        _set_span_attribute(
            span, f"{prefix}.finish_reason", choice.get("finish_reason")
        )

        message = choice.get("message")
        if not message:
            return

        _set_span_attribute(span, f"{prefix}.role", message.get("role"))
        _set_span_attribute(span, f"{prefix}.content", message.get("content"))

        function_call = message.get("function_call")
        if not function_call:
            return

        _set_span_attribute(
            span, f"{prefix}.function_call.name", function_call.get("name")
        )
        _set_span_attribute(
            span, f"{prefix}.function_call.arguments", function_call.get("arguments")
        )


def _build_from_streaming_response(span, response):
    complete_response = {"choices": [], "model": ""}
    for item in response:
        item_to_yield = item
        _accumulate_stream_items(item, complete_response)

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
        _accumulate_stream_items(item, complete_response)

        yield item_to_yield

    _set_response_attributes(span, complete_response)

    if should_send_prompts():
        _set_completions(span, complete_response.get("choices"))

    span.set_status(Status(StatusCode.OK))
    span.end()


def _accumulate_stream_items(item, complete_response):
    if is_openai_v1():
        item = model_as_dict(item)

    for choice in item.get("choices"):
        index = choice.get("index")
        if len(complete_response.get("choices")) <= index:
            complete_response["choices"].append(
                {"index": index, "message": {"content": "", "role": ""}}
            )
        complete_choice = complete_response.get("choices")[index]
        if choice.get("finish_reason"):
            complete_choice["finish_reason"] = choice.get("finish_reason")

        delta = choice.get("delta")

        if delta.get("content"):
            complete_choice["message"]["content"] += delta.get("content")
        if delta.get("role"):
            complete_choice["message"]["role"] = delta.get("role")
