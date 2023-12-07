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

from opentelemetry.trace import get_tracer, SpanKind
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.openai.utils import is_openai_v1

SPAN_NAME = "openai.chat"
LLM_REQUEST_TYPE = LLMRequestTypeValues.CHAT

logger = logging.getLogger(__name__)


@_with_tracer_wrapper
def chat_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    with tracer.start_as_current_span(SPAN_NAME, kind=SpanKind.CLIENT) as span:
        _set_request_attributes(span, LLM_REQUEST_TYPE, kwargs)
        if should_send_prompts():
            _set_prompts(span, kwargs.get("messages"))
            _set_functions_attributes(span, kwargs.get("functions"))

        response = wrapped(*args, **kwargs)

        print(response)

        if is_streaming_response(response):
            # TODO: WTH is this?
            _build_from_streaming_response(span, LLM_REQUEST_TYPE, response)
        else:
            if is_openai_v1():
                response = response.__dict__

            _set_response_attributes(span, response)
            
        if should_send_prompts():
            _set_completions(span, response.get("choices"))

        return response


@_with_tracer_wrapper
async def achat_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    async with start_as_current_span_async(tracer=tracer, name=SPAN_NAME, kind=SpanKind.CLIENT) as span:
        _set_request_attributes(span, LLM_REQUEST_TYPE, kwargs)
        if should_send_prompts():
            _set_span_prompts(span, kwargs.get("messages"))
            _set_functions_attributes(span, kwargs.get("functions"))

        response = await wrapped(*args, **kwargs)
        
        if is_streaming_response(response):
            # TODO: WTH is this?
            _build_from_streaming_response(span, LLM_REQUEST_TYPE, response)
        else:
            if is_openai_v1():
                response_dict = response.__dict__

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
            _set_span_attribute(span, f"{prefix}.role", msg.get("role"))
            _set_span_attribute(span, f"{prefix}.content", msg.get("content"))
    except Exception as ex:  # pylint: disable=broad-except
        logger.warning("Failed to set prompts for openai span, error: %s", str(ex))
        

def _set_completions(span, choices):
    if choices is None:
        return

    for choice in choices:
        if is_openai_v1() and not isinstance(choice, dict):
            choice = choice.__dict__

        index = choice.get("index")
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
        _set_span_attribute(span, f"{prefix}.finish_reason", choice.get("finish_reason"))

        message = choice.get("message")
        if not message:
            return

        if is_openai_v1() and not isinstance(message, dict):
            message = message.__dict__

        _set_span_attribute(span, f"{prefix}.role", message.get("role"))
        _set_span_attribute(span, f"{prefix}.content", message.get("content"))

        function_call = message.get("function_call")
        if not function_call:
            return

        if is_openai_v1() and not isinstance(function_call, dict):
            function_call = function_call.__dict__

        _set_span_attribute(span, f"{prefix}.function_call.name", function_call.get("name"))
        _set_span_attribute(span, f"{prefix}.function_call.arguments", function_call.get("arguments"))
