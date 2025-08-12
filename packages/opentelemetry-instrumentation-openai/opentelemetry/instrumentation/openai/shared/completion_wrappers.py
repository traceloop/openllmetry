import logging

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.instrumentation.openai.shared import (
    _set_client_attributes,
    _set_functions_attributes,
    _set_request_attributes,
    _set_response_attributes,
    _set_span_attribute,
    _set_span_stream_usage,
    is_streaming_response,
    model_as_dict,
    propagate_trace_context,
)
from opentelemetry.instrumentation.openai.shared.config import Config
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.instrumentation.openai.shared.event_emitter import emit_event
from opentelemetry.instrumentation.openai.shared.event_models import (
    ChoiceEvent,
    MessageEvent,
)
from opentelemetry.instrumentation.openai.utils import (
    _with_tracer_wrapper,
    dont_throw,
    is_openai_v1,
    should_emit_events,
    should_send_prompts,
)
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    SpanAttributes,
)
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import Status, StatusCode

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

    # Use the span as current context to ensure events get proper trace context
    with trace.use_span(span, end_on_exit=False):
        _handle_request(span, kwargs, instance)

        try:
            response = wrapped(*args, **kwargs)
        except Exception as e:
            span.set_attribute(ERROR_TYPE, e.__class__.__name__)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.end()
            raise

        if is_streaming_response(response):
            # span will be closed after the generator is done
            return _build_from_streaming_response(span, kwargs, response)
        else:
            _handle_response(response, span, instance)

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

    # Use the span as current context to ensure events get proper trace context
    with trace.use_span(span, end_on_exit=False):
        _handle_request(span, kwargs, instance)

        try:
            response = await wrapped(*args, **kwargs)
        except Exception as e:
            span.set_attribute(ERROR_TYPE, e.__class__.__name__)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.end()
            raise

        if is_streaming_response(response):
            # span will be closed after the generator is done
            return _abuild_from_streaming_response(span, kwargs, response)
        else:
            _handle_response(response, span, instance)

        span.end()
        return response


@dont_throw
def _handle_request(span, kwargs, instance):
    _set_request_attributes(span, kwargs, instance)
    if should_emit_events():
        _emit_prompts_events(kwargs)
    else:
        if should_send_prompts():
            _set_prompts(span, kwargs.get("prompt"))
            _set_functions_attributes(span, kwargs.get("functions"))
    _set_client_attributes(span, instance)
    if Config.enable_trace_context_propagation:
        propagate_trace_context(span, kwargs)


def _emit_prompts_events(kwargs):
    prompt = kwargs.get("prompt")
    if isinstance(prompt, list):
        for p in prompt:
            emit_event(MessageEvent(content=p))
    elif isinstance(prompt, str):
        emit_event(MessageEvent(content=prompt))


@dont_throw
def _handle_response(response, span, instance=None):
    if is_openai_v1():
        response_dict = model_as_dict(response)
    else:
        response_dict = response

    _set_response_attributes(span, response_dict)
    if should_emit_events():
        for choice in response.choices:
            emit_event(_parse_choice_event(choice))
    else:
        if should_send_prompts():
            _set_completions(span, response_dict.get("choices"))


def _set_prompts(span, prompt):
    if not span.is_recording() or not prompt:
        return

    _set_span_attribute(
        span,
        f"{SpanAttributes.LLM_PROMPTS}.0.user",
        prompt[0] if isinstance(prompt, list) else prompt,
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

    if should_emit_events():
        _emit_streaming_response_events(complete_response)
    else:
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

    if should_emit_events():
        _emit_streaming_response_events(complete_response)
    else:
        if should_send_prompts():
            _set_completions(span, complete_response.get("choices"))

    span.set_status(Status(StatusCode.OK))
    span.end()


def _emit_streaming_response_events(complete_response):
    for i, choice in enumerate(complete_response["choices"]):
        emit_event(
            ChoiceEvent(
                index=choice.get("index", i),
                message={"content": choice.get("text"), "role": "assistant"},
                finish_reason=choice.get("finish_reason", "unknown"),
            )
        )


@dont_throw
def _set_token_usage(span, request_kwargs, complete_response):
    prompt_usage = -1
    completion_usage = -1

    # Use token usage from API response only
    if complete_response.get("usage"):
        usage = complete_response["usage"]
        if usage.get("prompt_tokens"):
            prompt_usage = usage["prompt_tokens"]
        if usage.get("completion_tokens"):
            completion_usage = usage["completion_tokens"]

    # span record
    _set_span_stream_usage(span, prompt_usage, completion_usage)


@dont_throw
def _accumulate_streaming_response(complete_response, item):
    if is_openai_v1():
        item = model_as_dict(item)

    complete_response["model"] = item.get("model")
    complete_response["id"] = item.get("id")

    # capture usage information from the stream chunks
    if item.get("usage"):
        complete_response["usage"] = item.get("usage")

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


def _parse_choice_event(choice) -> ChoiceEvent:
    has_message = choice.text is not None
    has_finish_reason = choice.finish_reason is not None

    content = choice.text if has_message else None
    finish_reason = choice.finish_reason if has_finish_reason else "unknown"

    return ChoiceEvent(
        index=choice.index,
        message={"content": content, "role": "assistant"},
        finish_reason=finish_reason,
    )
