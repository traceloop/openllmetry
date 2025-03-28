import logging

from opentelemetry import context as context_api

from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
    LLMRequestTypeValues,
)

from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.instrumentation.openai.utils import _with_tracer_wrapper, dont_throw
from opentelemetry.instrumentation.openai.shared import (
    _set_client_attributes,
    _set_request_attributes,
    _set_span_attribute,
    _set_functions_attributes,
    _set_response_attributes,
    is_streaming_response,
    should_send_prompts,
    model_as_dict,
    should_record_stream_token_usage,
    get_token_count_from_string,
    _set_span_stream_usage,
    propagate_trace_context,
)

from opentelemetry.instrumentation.openai.utils import is_openai_v1

from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.openai.shared.config import Config

SPAN_NAME = "openai.completion"
LLM_REQUEST_TYPE = LLMRequestTypeValues.COMPLETION

logger = logging.getLogger(__name__)


@_with_tracer_wrapper
def completion_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    # span needs to be opened and closed manually because the response is a generator
    span = tracer.start_span(
        SPAN_NAME,
        kind=SpanKind.CLIENT,
        attributes={SpanAttributes.LLM_REQUEST_TYPE: LLM_REQUEST_TYPE.value},
    )

    _handle_request(span, kwargs, instance)
    try:
        response = wrapped(*args, **kwargs)
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise e

    if is_streaming_response(response):
        # span will be closed after the generator is done
        return _build_from_streaming_response(span, kwargs, response)
    else:
        _handle_response(response, span)

    span.end()
    return response


@_with_tracer_wrapper
async def acompletion_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return await wrapped(*args, **kwargs)

    span = tracer.start_span(
        name=SPAN_NAME,
        kind=SpanKind.CLIENT,
        attributes={SpanAttributes.LLM_REQUEST_TYPE: LLM_REQUEST_TYPE.value},
    )

    _handle_request(span, kwargs, instance)
    try:
        response = await wrapped(*args, **kwargs)
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise e

    if is_streaming_response(response):
        # span will be closed after the generator is done
        return _abuild_from_streaming_response(span, kwargs, response)
    else:
        _handle_response(response, span)

    span.end()
    return response


@_with_tracer_wrapper
def batch_completion_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    span = tracer.start_span(
        "openai.batch_completion",
        kind=SpanKind.CLIENT,
        attributes={SpanAttributes.LLM_REQUEST_TYPE: LLM_REQUEST_TYPE.value},
    )

    _handle_request(span, kwargs, instance)
    try:
        response = wrapped(*args, **kwargs)
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise e

    _handle_response(response, span)
    span.end()
    return response


@_with_tracer_wrapper
async def abatch_completion_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return await wrapped(*args, **kwargs)

    span = tracer.start_span(
        "openai.batch_completion",
        kind=SpanKind.CLIENT,
        attributes={SpanAttributes.LLM_REQUEST_TYPE: LLM_REQUEST_TYPE.value},
    )

    _handle_request(span, kwargs, instance)
    try:
        response = await wrapped(*args, **kwargs)
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise e

    _handle_response(response, span)
    span.end()
    return response


@dont_throw
def _handle_request(span, kwargs, instance):
    _set_request_attributes(span, kwargs)
    if should_send_prompts():
        _set_prompts(span, kwargs.get("prompt"))
        _set_functions_attributes(span, kwargs.get("functions"))
    _set_client_attributes(span, instance)
    if Config.enable_trace_context_propagation:
        propagate_trace_context(span, kwargs)


@dont_throw
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

    if isinstance(prompt, list):
        for i, p in enumerate(prompt):
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.{i}.user",
                p
            )
    else:
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_PROMPTS}.0.user",
            prompt
        )


@dont_throw
def _set_completions(span, choices):
    if not span.is_recording() or not choices:
        return

    for choice in choices:
        index = choice.get("index")
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
        _set_span_attribute(
            span, f"{prefix}.finish_reason", choice.get("finish_reason")
        )
        _set_span_attribute(span, f"{prefix}.content", choice.get("text"))


@dont_throw
def _build_from_streaming_response(span, request_kwargs, response):
    complete_response = {"choices": [], "model": "", "id": ""}
    for item in response:
        yield item
        _accumulate_streaming_response(complete_response, item)

    _set_response_attributes(span, complete_response)

    _set_token_usage(span, request_kwargs, complete_response)

    if should_send_prompts():
        _set_completions(span, complete_response.get("choices"))

    span.set_status(Status(StatusCode.OK))
    span.end()


@dont_throw
async def _abuild_from_streaming_response(span, request_kwargs, response):
    complete_response = {"choices": [], "model": "", "id": ""}
    async for item in response:
        yield item
        _accumulate_streaming_response(complete_response, item)

    _set_response_attributes(span, complete_response)

    _set_token_usage(span, request_kwargs, complete_response)

    if should_send_prompts():
        _set_completions(span, complete_response.get("choices"))

    span.set_status(Status(StatusCode.OK))
    span.end()


@dont_throw 
def _set_token_usage(span, request_kwargs, complete_response):
    # Initialize with zeros to ensure we always have valid counts
    prompt_usage = 0
    completion_usage = 0

    # prompt_usage calculation 
    if request_kwargs and request_kwargs.get("prompt"):
        prompt_content = request_kwargs.get("prompt")
        model_name = complete_response.get("model") or request_kwargs.get("model") or "davinci-002"

        if isinstance(prompt_content, list):
            for p in prompt_content:
                p_count = get_token_count_from_string(str(p), model_name)
                if p_count is None:  # Fallback if token counting fails
                    p_count = len(str(p).split()) * 2
                prompt_usage += p_count
        else:
            prompt_usage = get_token_count_from_string(str(prompt_content), model_name)
            if prompt_usage is None:  # Fallback if token counting fails
                prompt_usage = len(str(prompt_content).split()) * 2

    # completion_usage calculation
    completion_content = ""
    if complete_response.get("choices"):
        model_name = complete_response.get("model") or request_kwargs.get("model") or "davinci-002"

        for choice in complete_response.get("choices"):
            if choice.get("text"):
                completion_content += str(choice.get("text"))

    if completion_content:
        completion_usage = get_token_count_from_string(completion_content, model_name)
        if completion_usage is None:  # Fallback if token counting fails
            completion_usage = len(completion_content.split()) * 2

    # Always set span usage metrics since we have valid counts
    _set_span_stream_usage(span, prompt_usage, completion_usage)


@dont_throw
def _accumulate_streaming_response(complete_response, item):
    if is_openai_v1():
        item = model_as_dict(item)

    complete_response["model"] = item.get("model")
    complete_response["id"] = item.get("id")
    for choice in item.get("choices"):
        index = choice.get("index")
        if len(complete_response.get("choices")) <= index:
            complete_response["choices"].append({"index": index, "text": ""})
        complete_choice = complete_response.get("choices")[index]
        if choice.get("finish_reason"):
            complete_choice["finish_reason"] = choice.get("finish_reason")

        if choice.get("text"):
            complete_choice["text"] += choice.get("text")

    return complete_response
