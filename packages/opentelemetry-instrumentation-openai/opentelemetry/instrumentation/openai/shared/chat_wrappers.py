import json
import logging
import time
from wrapt import ObjectProxy


from opentelemetry import context as context_api
from opentelemetry.metrics import Counter, Histogram
from opentelemetry.semconv.ai import SpanAttributes, LLMRequestTypeValues

from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.instrumentation.openai.utils import (
    _with_chat_telemetry_wrapper,
    dont_throw,
)
from opentelemetry.instrumentation.openai.shared import (
    _metric_shared_attributes,
    _set_client_attributes,
    _set_request_attributes,
    _set_span_attribute,
    _set_functions_attributes,
    _token_type,
    set_tools_attributes,
    _set_response_attributes,
    is_streaming_response,
    should_send_prompts,
    model_as_dict,
    _get_openai_base_url,
    OPENAI_LLM_USAGE_TOKEN_TYPES,
    should_record_stream_token_usage,
    get_token_count_from_string,
    _set_span_stream_usage,
)
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.openai.utils import is_openai_v1, is_azure_openai

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
        if is_openai_v1():
            return ChatStream(
                span,
                response,
                instance,
                token_counter,
                choice_counter,
                duration_histogram,
                streaming_time_to_first_token,
                streaming_time_to_generate,
                start_time,
                kwargs,
            )
        else:
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
                kwargs,
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


@_with_chat_telemetry_wrapper
async def achat_wrapper(
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

    span = tracer.start_span(
        SPAN_NAME,
        kind=SpanKind.CLIENT,
        attributes={SpanAttributes.LLM_REQUEST_TYPE: LLM_REQUEST_TYPE.value},
    )
    _handle_request(span, kwargs, instance)

    try:
        start_time = time.time()
        response = await wrapped(*args, **kwargs)
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
        if is_openai_v1():
            return ChatStream(
                span,
                response,
                instance,
                token_counter,
                choice_counter,
                duration_histogram,
                streaming_time_to_first_token,
                streaming_time_to_generate,
                start_time,
                kwargs,
            )
        else:
            return _abuild_from_streaming_response(
                span,
                response,
                instance,
                token_counter,
                choice_counter,
                duration_histogram,
                streaming_time_to_first_token,
                streaming_time_to_generate,
                start_time,
                kwargs,
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


@dont_throw
def _handle_request(span, kwargs, instance):
    _set_request_attributes(span, kwargs)
    _set_client_attributes(span, instance)
    if should_send_prompts():
        _set_prompts(span, kwargs.get("messages"))
        if kwargs.get("functions"):
            _set_functions_attributes(span, kwargs.get("functions"))
        elif kwargs.get("tools"):
            set_tools_attributes(span, kwargs.get("tools"))


@dont_throw
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
    shared_attributes = _metric_shared_attributes(
        response_model=response_dict.get("model") or None,
        operation="chat",
        server_address=_get_openai_base_url(instance),
        is_streaming=False,
    )

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
                "gen_ai.token.type": _token_type(name),
            }
            token_counter.record(val, attributes=attributes_with_token_type)


def _set_prompts(span, messages):
    if not span.is_recording() or messages is None:
        return

    for i, msg in enumerate(messages):
        prefix = f"{SpanAttributes.LLM_PROMPTS}.{i}"
        if isinstance(msg.get("content"), str):
            content = msg.get("content")
        elif isinstance(msg.get("content"), list):
            content = json.dumps(msg.get("content"))

        _set_span_attribute(span, f"{prefix}.role", msg.get("role"))
        _set_span_attribute(span, f"{prefix}.content", content)


def _set_completions(span, choices):
    if choices is None:
        return

    for choice in choices:
        index = choice.get("index")
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
        _set_span_attribute(
            span, f"{prefix}.finish_reason", choice.get("finish_reason")
        )

        if choice.get("finish_reason") == "content_filter":
            _set_span_attribute(span, f"{prefix}.role", "assistant")
            _set_span_attribute(span, f"{prefix}.content", "FILTERED")
            return

        message = choice.get("message")
        if not message:
            return

        _set_span_attribute(span, f"{prefix}.role", message.get("role"))
        _set_span_attribute(span, f"{prefix}.content", message.get("content"))

        function_call = message.get("function_call")
        if function_call:
            _set_span_attribute(
                span, f"{prefix}.function_call.name", function_call.get("name")
            )
            _set_span_attribute(
                span,
                f"{prefix}.function_call.arguments",
                function_call.get("arguments"),
            )

        tool_calls = message.get("tool_calls")
        if tool_calls:
            _set_span_attribute(
                span,
                f"{prefix}.function_call.name",
                tool_calls[0].get("function").get("name"),
            )
            _set_span_attribute(
                span,
                f"{prefix}.function_call.arguments",
                tool_calls[0].get("function").get("arguments"),
            )


def _set_streaming_token_metrics(
    request_kwargs, complete_response, span, token_counter, shared_attributes
):
    # use tiktoken calculate token usage
    if not should_record_stream_token_usage():
        return

    # kwargs={'model': 'gpt-3.5', 'messages': [{'role': 'user', 'content': '...'}], 'stream': True}
    prompt_usage = -1
    completion_usage = -1

    # prompt_usage
    if request_kwargs and request_kwargs.get("messages"):
        prompt_content = ""
        # setting the default model_name as gpt-4. As this uses the embedding "cl100k_base" that
        # is used by most of the other model.
        model_name = (
            request_kwargs.get("model") or complete_response.get("model") or "gpt-4"
        )
        for msg in request_kwargs.get("messages"):
            if msg.get("content"):
                prompt_content += msg.get("content")
        if model_name:
            prompt_usage = get_token_count_from_string(prompt_content, model_name)

    # completion_usage
    if complete_response.get("choices"):
        completion_content = ""
        # setting the default model_name as gpt-4. As this uses the embedding "cl100k_base" that
        # is used by most of the other model.
        model_name = complete_response.get("model") or "gpt-4"

        for choice in complete_response.get("choices"):
            if choice.get("message") and choice.get("message").get("content"):
                completion_content += choice["message"]["content"]

        if model_name:
            completion_usage = get_token_count_from_string(
                completion_content, model_name
            )

    # span record
    _set_span_stream_usage(span, prompt_usage, completion_usage)

    # metrics record
    if token_counter:
        if type(prompt_usage) is int and prompt_usage >= 0:
            attributes_with_token_type = {
                **shared_attributes,
                "gen_ai.token.type": "input",
            }
            token_counter.record(prompt_usage, attributes=attributes_with_token_type)

        if type(completion_usage) is int and completion_usage >= 0:
            attributes_with_token_type = {
                **shared_attributes,
                "gen_ai.token.type": "output",
            }
            token_counter.record(
                completion_usage, attributes=attributes_with_token_type
            )


class ChatStream(ObjectProxy):
    _span = None
    _instance = None
    _token_counter = None
    _choice_counter = None
    _duration_histogram = None
    _streaming_time_to_first_token = None
    _streaming_time_to_generate = None
    _start_time = None
    _request_kwargs = None

    def __init__(
        self,
        span,
        response,
        instance=None,
        token_counter=None,
        choice_counter=None,
        duration_histogram=None,
        streaming_time_to_first_token=None,
        streaming_time_to_generate=None,
        start_time=None,
        request_kwargs=None,
    ):
        super().__init__(response)

        print("HEYY", response.__class__.__name__)

        self._span = span
        self._instance = instance
        self._token_counter = token_counter
        self._choice_counter = choice_counter
        self._duration_histogram = duration_histogram
        self._streaming_time_to_first_token = streaming_time_to_first_token
        self._streaming_time_to_generate = streaming_time_to_generate
        self._start_time = start_time
        self._request_kwargs = request_kwargs

        self._first_token = True
        # will be updated when first token is received
        self._time_of_first_token = self._start_time
        self._complete_response = {"choices": [], "model": ""}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__wrapped__.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.__wrapped__.__aexit__(exc_type, exc_val, exc_tb)

    def __iter__(self):
        return self

    def __aiter__(self):
        return self

    def __next__(self):
        try:
            chunk = self.__wrapped__.__next__()
        except Exception as e:
            if isinstance(e, StopIteration):
                self._close_span()
            raise e
        else:
            self._process_item(chunk)
            return chunk

    async def __anext__(self):
        try:
            chunk = await self.__wrapped__.__anext__()
        except Exception as e:
            if isinstance(e, StopAsyncIteration):
                self._close_span()
            raise e
        else:
            self._process_item(chunk)
            return chunk

    def _process_item(self, item):
        self._span.add_event(name="llm.content.completion.chunk")

        if self._first_token and self._streaming_time_to_first_token:
            self._time_of_first_token = time.time()
            self._streaming_time_to_first_token.record(
                self._time_of_first_token - self._start_time,
                attributes=self._shared_attributes(),
            )
            self._first_token = False

        if is_openai_v1():
            item = model_as_dict(item)

        self._complete_response["model"] = item.get("model")

        for choice in item.get("choices"):
            index = choice.get("index")
            if len(self._complete_response.get("choices")) <= index:
                self._complete_response["choices"].append(
                    {"index": index, "message": {"content": "", "role": ""}}
                )
            complete_choice = self._complete_response.get("choices")[index]
            if choice.get("finish_reason"):
                complete_choice["finish_reason"] = choice.get("finish_reason")

            delta = choice.get("delta")

            if delta and delta.get("content"):
                complete_choice["message"]["content"] += delta.get("content")
            if delta and delta.get("role"):
                complete_choice["message"]["role"] = delta.get("role")

    def _shared_attributes(self):
        return _metric_shared_attributes(
            response_model=self._complete_response.get("model")
            or self._request_kwargs.get("model")
            or None,
            operation="chat",
            server_address=_get_openai_base_url(self._instance),
            is_streaming=True,
        )

    def _close_span(self):
        if not is_azure_openai(self._instance):
            _set_streaming_token_metrics(
                self._request_kwargs,
                self._complete_response,
                self._span,
                self._token_counter,
                self._shared_attributes(),
            )

        # choice metrics
        if self._choice_counter and self._complete_response.get("choices"):
            _set_choice_counter_metrics(
                self._choice_counter,
                self._complete_response.get("choices"),
                self._shared_attributes(),
            )

        # duration metrics
        if self._start_time and isinstance(self._start_time, (float, int)):
            duration = time.time() - self._start_time
        else:
            duration = None
        if duration and isinstance(duration, (float, int)) and self._duration_histogram:
            self._duration_histogram.record(
                duration, attributes=self._shared_attributes()
            )
        if self._streaming_time_to_generate and self._time_of_first_token:
            self._streaming_time_to_generate.record(
                time.time() - self._time_of_first_token,
                attributes=self._shared_attributes(),
            )

        _set_response_attributes(self._span, self._complete_response)

        if should_send_prompts():
            _set_completions(self._span, self._complete_response.get("choices"))

        self._span.set_status(Status(StatusCode.OK))
        self._span.end()


# Backward compatibility with OpenAI v0


@dont_throw
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
    request_kwargs=None,
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
        "gen_ai.response.model": complete_response.get("model") or None,
        "server.address": _get_openai_base_url(instance),
        "stream": True,
    }

    if not is_azure_openai(instance):
        _set_streaming_token_metrics(
            request_kwargs, complete_response, span, token_counter, shared_attributes
        )

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


@dont_throw
async def _abuild_from_streaming_response(
    span,
    response,
    instance=None,
    token_counter=None,
    choice_counter=None,
    duration_histogram=None,
    streaming_time_to_first_token=None,
    streaming_time_to_generate=None,
    start_time=None,
    request_kwargs=None,
):
    complete_response = {"choices": [], "model": ""}

    first_token = True
    time_of_first_token = start_time  # will be updated when first token is received

    async for item in response:
        span.add_event(name="llm.content.completion.chunk")

        item_to_yield = item

        if first_token and streaming_time_to_first_token:
            time_of_first_token = time.time()
            streaming_time_to_first_token.record(time_of_first_token - start_time)
            first_token = False

        _accumulate_stream_items(item, complete_response)

        yield item_to_yield

    shared_attributes = {
        "gen_ai.response.model": complete_response.get("model") or None,
        "server.address": _get_openai_base_url(instance),
        "stream": True,
    }

    if not is_azure_openai(instance):
        _set_streaming_token_metrics(
            request_kwargs, complete_response, span, token_counter, shared_attributes
        )

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

        if delta and delta.get("content"):
            complete_choice["message"]["content"] += delta.get("content")
        if delta and delta.get("role"):
            complete_choice["message"]["role"] = delta.get("role")
