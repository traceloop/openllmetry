import json
import logging
import time

from opentelemetry import context as context_api
from opentelemetry.metrics import Counter, Histogram
from opentelemetry.semconv.ai import SpanAttributes, LLMRequestTypeValues

from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.instrumentation.openai.utils import (
    _with_tracer_wrapper,
    _with_chat_telemetry_wrapper,
)
from opentelemetry.instrumentation.openai.shared import (
    _set_client_attributes,
    _set_request_attributes,
    _set_span_attribute,
    _set_functions_attributes,
    _set_response_attributes,
    is_streaming_response,
    should_send_prompts,
    model_as_dict,
    _get_openai_base_url,
    OPENAI_LLM_USAGE_TOKEN_TYPES,
)
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.openai.utils import is_openai_v1

SPAN_NAME = "openai.chat"
LLM_REQUEST_TYPE = LLMRequestTypeValues.CHAT

logger = logging.getLogger(__name__)


@_with_chat_telemetry_wrapper
def chat_wrapper(
    tracer: Tracer,
    token_counter: Counter,
    choice_counter: Counter,
    duration_histogram: Histogram,
    exception_counter: Counter,
    streaming_time_to_first_token: Histogram,
    streaming_time_to_generate: Histogram,
    wrapped,
    instance,
    args,
    kwargs,
):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    # span needs to be opened and closed manually because the response is a generator
    span = tracer.start_span(
        SPAN_NAME,
        kind=SpanKind.CLIENT,
        attributes={SpanAttributes.LLM_REQUEST_TYPE: LLM_REQUEST_TYPE.value},
    )

    _handle_request(span, kwargs, instance)

    try:
        start_time = time.time()
        response = wrapped(*args, **kwargs)
        end_time = time.time()
    except Exception as e:  # pylint: disable=broad-except
        end_time = time.time()
        duration = end_time - start_time if "start_time" in locals() else 0

        attributes = {
            "error.type": e.__class__.__name__,
        }

        if duration > 0 and duration_histogram:
            duration_histogram.record(duration, attributes=attributes)
        if exception_counter:
            exception_counter.add(1, attributes=attributes)

        raise e

    if is_streaming_response(response):
        # span will be closed after the generator is done
        return _build_from_streaming_response(
            span,
            response,
            instance,
            token_counter,
            choice_counter,
            duration_histogram,
            streaming_time_to_first_token,
            streaming_time_to_generate,
            start_time,
        )

    duration = end_time - start_time

    _handle_response(
        response,
        span,
        instance,
        token_counter,
        choice_counter,
        duration_histogram,
        duration,
    )
    span.end()

    return response


@_with_tracer_wrapper
async def achat_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    span = tracer.start_span(
        SPAN_NAME,
        kind=SpanKind.CLIENT,
        attributes={SpanAttributes.LLM_REQUEST_TYPE: LLM_REQUEST_TYPE.value},
    )
    _handle_request(span, kwargs, instance)
    response = await wrapped(*args, **kwargs)

    if is_streaming_response(response):
        # span will be closed after the generator is done
        return _abuild_from_streaming_response(span, response)

    _handle_response(response, span)
    span.end()

    return response


def _handle_request(span, kwargs, instance):
    _set_request_attributes(span, kwargs)
    _set_client_attributes(span, instance)
    if should_send_prompts():
        _set_prompts(span, kwargs.get("messages"))
        _set_functions_attributes(span, kwargs.get("functions"))


def _handle_response(
    response,
    span,
    instance=None,
    token_counter=None,
    choice_counter=None,
    duration_histogram=None,
    duration=None,
):
    if is_openai_v1():
        response_dict = model_as_dict(response)
    else:
        response_dict = response

    # metrics record
    _set_chat_metrics(
        instance,
        token_counter,
        choice_counter,
        duration_histogram,
        response_dict,
        duration,
    )

    # span attributes
    _set_response_attributes(span, response_dict)

    if should_send_prompts():
        _set_completions(span, response_dict.get("choices"))

    return response


def _set_chat_metrics(
    instance, token_counter, choice_counter, duration_histogram, response_dict, duration
):
    shared_attributes = {
        "llm.response.model": response_dict.get("model") or None,
        "server.address": _get_openai_base_url(instance),
    }

    # token metrics
    usage = response_dict.get("usage")  # type: dict
    if usage and token_counter:
        _set_token_counter_metrics(token_counter, usage, shared_attributes)

    # choices metrics
    choices = response_dict.get("choices")
    if choices and choice_counter:
        _set_choice_counter_metrics(choice_counter, choices, shared_attributes)

    # duration metrics
    if duration and isinstance(duration, (float, int)) and duration_histogram:
        duration_histogram.record(duration, attributes=shared_attributes)


def _set_choice_counter_metrics(choice_counter, choices, shared_attributes):
    choice_counter.add(len(choices), attributes=shared_attributes)
    for choice in choices:
        attributes_with_reason = {
            **shared_attributes,
            "llm.response.finish_reason": choice["finish_reason"],
        }
        choice_counter.add(1, attributes=attributes_with_reason)


def _set_token_counter_metrics(token_counter, usage, shared_attributes):
    for name, val in usage.items():
        if name in OPENAI_LLM_USAGE_TOKEN_TYPES:
            attributes_with_token_type = {
                **shared_attributes,
                "llm.usage.token_type": name.split("_")[0],
            }
            token_counter.add(val, attributes=attributes_with_token_type)


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


def _build_from_streaming_response(
    span,
    response,
    instance=None,
    token_counter=None,
    choice_counter=None,
    duration_histogram=None,
    streaming_time_to_first_token=None,
    streaming_time_to_generate=None,
    start_time=None,
):
    complete_response = {"choices": [], "model": ""}

    first_token = True
    time_of_first_token = start_time  # will be updated when first token is received

    for item in response:
        span.add_event(name="llm.content.completion.chunk")

        item_to_yield = item

        if first_token and streaming_time_to_first_token:
            time_of_first_token = time.time()
            streaming_time_to_first_token.record(time_of_first_token - start_time)
            first_token = False

        _accumulate_stream_items(item, complete_response)

        yield item_to_yield

    shared_attributes = {
        "llm.response.model": complete_response.get("model") or None,
        "server.address": _get_openai_base_url(instance),
        "stream": True,
    }

    # can't get token usage in stream mode
    # choice metrics
    if choice_counter and complete_response.get("choices"):
        _set_choice_counter_metrics(
            choice_counter, complete_response.get("choices"), shared_attributes
        )

    # duration metrics
    if start_time and isinstance(start_time, (float, int)):
        duration = time.time() - start_time
    else:
        duration = None
    if duration and isinstance(duration, (float, int)) and duration_histogram:
        duration_histogram.record(duration, attributes=shared_attributes)
    if streaming_time_to_generate and time_of_first_token:
        streaming_time_to_generate.record(time.time() - time_of_first_token)

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

    complete_response["model"] = item.get("model")

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
