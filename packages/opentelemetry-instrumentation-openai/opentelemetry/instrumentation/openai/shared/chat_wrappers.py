import copy
import json
import logging
import threading
import time
from functools import singledispatch
from typing import List, Optional, Union

from opentelemetry import context as context_api
import pydantic
from opentelemetry.instrumentation.openai.shared import (
    OPENAI_LLM_USAGE_TOKEN_TYPES,
    _get_openai_base_url,
    _set_client_attributes,
    _set_functions_attributes,
    _set_request_attributes,
    _set_response_attributes,
    _set_span_attribute,
    _set_span_stream_usage,
    _token_type,
    is_streaming_response,
    metric_shared_attributes,
    model_as_dict,
    propagate_trace_context,
    set_tools_attributes,
)
from opentelemetry.instrumentation.openai.shared.config import Config
from opentelemetry.instrumentation.openai.shared.event_emitter import emit_event
from opentelemetry.instrumentation.openai.shared.event_models import (
    ChoiceEvent,
    MessageEvent,
    ToolCall,
)
from opentelemetry.instrumentation.openai.utils import (
    _with_chat_telemetry_wrapper,
    dont_throw,
    is_openai_v1,
    run_async,
    should_emit_events,
    should_send_prompts,
)
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.metrics import Counter, Histogram
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    SpanAttributes,
)
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry import trace
from opentelemetry.trace.status import Status, StatusCode
from wrapt import ObjectProxy

SPAN_NAME = "openai.chat"
PROMPT_FILTER_KEY = "prompt_filter_results"
CONTENT_FILTER_KEY = "content_filter_results"

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
        run_async(_handle_request(span, kwargs, instance))
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

            span.set_attribute(ERROR_TYPE, e.__class__.__name__)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.end()

            raise

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
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return await wrapped(*args, **kwargs)

    span = tracer.start_span(
        SPAN_NAME,
        kind=SpanKind.CLIENT,
        attributes={SpanAttributes.LLM_REQUEST_TYPE: LLM_REQUEST_TYPE.value},
    )

    # Use the span as current context to ensure events get proper trace context
    with trace.use_span(span, end_on_exit=False):
        await _handle_request(span, kwargs, instance)

        try:
            start_time = time.time()
            response = await wrapped(*args, **kwargs)
            end_time = time.time()
        except Exception as e:  # pylint: disable=broad-except
            end_time = time.time()
            duration = end_time - start_time if "start_time" in locals() else 0

            common_attributes = Config.get_common_metrics_attributes()
            attributes = {
                **common_attributes,
                "error.type": e.__class__.__name__,
            }

            if duration > 0 and duration_histogram:
                duration_histogram.record(duration, attributes=attributes)
            if exception_counter:
                exception_counter.add(1, attributes=attributes)

            span.set_attribute(ERROR_TYPE, e.__class__.__name__)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.end()

            raise

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
async def _handle_request(span, kwargs, instance):
    _set_request_attributes(span, kwargs, instance)
    _set_client_attributes(span, instance)
    if should_emit_events():
        for message in kwargs.get("messages", []):
            emit_event(
                MessageEvent(
                    content=message.get("content"),
                    role=message.get("role"),
                    tool_calls=_parse_tool_calls(
                        message.get("tool_calls", None)),
                )
            )
    else:
        if should_send_prompts():
            await _set_prompts(span, kwargs.get("messages"))
            if kwargs.get("functions"):
                _set_functions_attributes(span, kwargs.get("functions"))
            elif kwargs.get("tools"):
                set_tools_attributes(span, kwargs.get("tools"))
    if Config.enable_trace_context_propagation:
        propagate_trace_context(span, kwargs)

    # Reasoning request attributes
    reasoning_effort = kwargs.get("reasoning_effort")
    _set_span_attribute(
        span,
        SpanAttributes.LLM_REQUEST_REASONING_EFFORT,
        reasoning_effort or ()
    )


@dont_throw
def _handle_response(
    response,
    span,
    instance=None,
    token_counter=None,
    choice_counter=None,
    duration_histogram=None,
    duration=None,
    is_streaming: bool = False,
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
        is_streaming,
    )

    # span attributes
    _set_response_attributes(span, response_dict)

    # Reasoning usage attributes
    usage = response_dict.get("usage")
    reasoning_tokens = None
    if usage:
        # Support both dict-style and object-style `usage`
        tokens_details = (
            usage.get("completion_tokens_details") if isinstance(usage, dict)
            else getattr(usage, "completion_tokens_details", None)
        )

        if tokens_details:
            reasoning_tokens = (
                tokens_details.get("reasoning_tokens", None) if isinstance(tokens_details, dict)
                else getattr(tokens_details, "reasoning_tokens", None)
            )

    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_REASONING_TOKENS,
        reasoning_tokens or 0,
    )

    if should_emit_events():
        if response.choices is not None:
            for choice in response.choices:
                emit_event(_parse_choice_event(choice))
    else:
        if should_send_prompts():
            _set_completions(span, response_dict.get("choices"))

    return response


def _set_chat_metrics(
    instance,
    token_counter,
    choice_counter,
    duration_histogram,
    response_dict,
    duration,
    is_streaming: bool = False,
):
    shared_attributes = metric_shared_attributes(
        response_model=response_dict.get("model") or None,
        operation="chat",
        server_address=_get_openai_base_url(instance),
        is_streaming=is_streaming,
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
        attributes_with_reason = {**shared_attributes}
        if choice.get("finish_reason"):
            attributes_with_reason[SpanAttributes.LLM_RESPONSE_FINISH_REASON] = (
                choice.get("finish_reason")
            )
        choice_counter.add(1, attributes=attributes_with_reason)


def _set_token_counter_metrics(token_counter, usage, shared_attributes):
    for name, val in usage.items():
        if name in OPENAI_LLM_USAGE_TOKEN_TYPES:
            attributes_with_token_type = {
                **shared_attributes,
                SpanAttributes.LLM_TOKEN_TYPE: _token_type(name),
            }
            token_counter.record(val, attributes=attributes_with_token_type)


def _is_base64_image(item):
    if not isinstance(item, dict):
        return False

    if not isinstance(item.get("image_url"), dict):
        return False

    if "data:image/" not in item.get("image_url", {}).get("url", ""):
        return False

    return True


async def _process_image_item(item, trace_id, span_id, message_index, content_index):
    if not Config.upload_base64_image:
        return item

    image_format = item["image_url"]["url"].split(";")[0].split("/")[1]
    image_name = f"message_{message_index}_content_{content_index}.{image_format}"
    base64_string = item["image_url"]["url"].split(",")[1]
    url = await Config.upload_base64_image(trace_id, span_id, image_name, base64_string)

    return {"type": "image_url", "image_url": {"url": url}}


@dont_throw
async def _set_prompts(span, messages):
    if not span.is_recording() or messages is None:
        return

    for i, msg in enumerate(messages):
        prefix = f"{SpanAttributes.LLM_PROMPTS}.{i}"
        msg = msg if isinstance(msg, dict) else model_as_dict(msg)

        _set_span_attribute(span, f"{prefix}.role", msg.get("role"))
        if msg.get("content"):
            content = copy.deepcopy(msg.get("content"))
            if isinstance(content, list):
                content = [
                    (
                        await _process_image_item(
                            item, span.context.trace_id, span.context.span_id, i, j
                        )
                        if _is_base64_image(item)
                        else item
                    )
                    for j, item in enumerate(content)
                ]

                content = json.dumps(content)
            _set_span_attribute(span, f"{prefix}.content", content)
        if msg.get("tool_call_id"):
            _set_span_attribute(
                span, f"{prefix}.tool_call_id", msg.get("tool_call_id"))
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for i, tool_call in enumerate(tool_calls):
                if is_openai_v1():
                    tool_call = model_as_dict(tool_call)

                function = tool_call.get("function")
                _set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.id",
                    tool_call.get("id"),
                )
                _set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.name",
                    function.get("name"),
                )
                _set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.arguments",
                    function.get("arguments"),
                )


def _set_completions(span, choices):
    if choices is None:
        return

    for choice in choices:
        index = choice.get("index")
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
        _set_span_attribute(
            span, f"{prefix}.finish_reason", choice.get("finish_reason")
        )

        if choice.get("content_filter_results"):
            _set_span_attribute(
                span,
                f"{prefix}.{CONTENT_FILTER_KEY}",
                json.dumps(choice.get("content_filter_results")),
            )

        if choice.get("finish_reason") == "content_filter":
            _set_span_attribute(span, f"{prefix}.role", "assistant")
            _set_span_attribute(span, f"{prefix}.content", "FILTERED")

            return

        message = choice.get("message")
        if not message:
            return

        _set_span_attribute(span, f"{prefix}.role", message.get("role"))

        if message.get("refusal"):
            _set_span_attribute(
                span, f"{prefix}.refusal", message.get("refusal"))
        else:
            _set_span_attribute(
                span, f"{prefix}.content", message.get("content"))

        function_call = message.get("function_call")
        if function_call:
            _set_span_attribute(
                span, f"{prefix}.tool_calls.0.name", function_call.get("name")
            )
            _set_span_attribute(
                span,
                f"{prefix}.tool_calls.0.arguments",
                function_call.get("arguments"),
            )

        tool_calls = message.get("tool_calls")
        if tool_calls:
            for i, tool_call in enumerate(tool_calls):
                function = tool_call.get("function")
                _set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.id",
                    tool_call.get("id"),
                )
                _set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.name",
                    function.get("name"),
                )
                _set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.arguments",
                    function.get("arguments"),
                )


@dont_throw
def _set_streaming_token_metrics(
    request_kwargs, complete_response, span, token_counter, shared_attributes
):
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

    # metrics record
    if token_counter:
        if isinstance(prompt_usage, int) and prompt_usage >= 0:
            attributes_with_token_type = {
                **shared_attributes,
                SpanAttributes.LLM_TOKEN_TYPE: "input",
            }
            token_counter.record(
                prompt_usage, attributes=attributes_with_token_type)

        if isinstance(completion_usage, int) and completion_usage >= 0:
            attributes_with_token_type = {
                **shared_attributes,
                SpanAttributes.LLM_TOKEN_TYPE: "output",
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

        # Cleanup state tracking to prevent duplicate operations
        self._cleanup_completed = False
        self._cleanup_lock = threading.Lock()

    def __del__(self):
        """Cleanup when object is garbage collected"""
        if hasattr(self, '_cleanup_completed') and not self._cleanup_completed:
            self._ensure_cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_exception = None
        try:
            self._ensure_cleanup()
        except Exception as e:
            cleanup_exception = e
            # Don't re-raise to avoid masking original exception

        result = self.__wrapped__.__exit__(exc_type, exc_val, exc_tb)

        if cleanup_exception:
            # Log cleanup exception but don't affect context manager behavior
            logger.debug(
                "Error during ChatStream cleanup in __exit__: %s", cleanup_exception)

        return result

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
                self._process_complete_response()
            else:
                # Handle cleanup for other exceptions during stream iteration
                self._ensure_cleanup()
                if self._span and self._span.is_recording():
                    self._span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        else:
            self._process_item(chunk)
            return chunk

    async def __anext__(self):
        try:
            chunk = await self.__wrapped__.__anext__()
        except Exception as e:
            if isinstance(e, StopAsyncIteration):
                self._process_complete_response()
            else:
                # Handle cleanup for other exceptions during stream iteration
                self._ensure_cleanup()
                if self._span and self._span.is_recording():
                    self._span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        else:
            self._process_item(chunk)
            return chunk

    def _process_item(self, item):
        self._span.add_event(
            name=f"{SpanAttributes.LLM_CONTENT_COMPLETION_CHUNK}")

        if self._first_token and self._streaming_time_to_first_token:
            self._time_of_first_token = time.time()
            self._streaming_time_to_first_token.record(
                self._time_of_first_token - self._start_time,
                attributes=self._shared_attributes(),
            )
            self._first_token = False

        _accumulate_stream_items(item, self._complete_response)

    def _shared_attributes(self):
        return metric_shared_attributes(
            response_model=self._complete_response.get("model")
            or self._request_kwargs.get("model")
            or None,
            operation="chat",
            server_address=_get_openai_base_url(self._instance),
            is_streaming=True,
        )

    @dont_throw
    def _process_complete_response(self):
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
        if should_emit_events():
            for choice in self._complete_response.get("choices", []):
                emit_event(_parse_choice_event(choice))
        else:
            if should_send_prompts():
                _set_completions(
                    self._span, self._complete_response.get("choices"))

        self._span.set_status(Status(StatusCode.OK))
        self._span.end()
        self._cleanup_completed = True

    @dont_throw
    def _ensure_cleanup(self):
        """Thread-safe cleanup method that handles different cleanup scenarios"""
        with self._cleanup_lock:
            if self._cleanup_completed:
                logger.debug("ChatStream cleanup already completed, skipping")
                return

            try:
                logger.debug("Starting ChatStream cleanup")

                # Calculate partial metrics based on available data
                self._record_partial_metrics()

                # Set span status and close it
                if self._span and self._span.is_recording():
                    self._span.set_status(Status(StatusCode.OK))
                    self._span.end()
                    logger.debug("ChatStream span closed successfully")

                self._cleanup_completed = True
                logger.debug("ChatStream cleanup completed successfully")

            except Exception as e:
                # Log cleanup errors but don't propagate to avoid masking original issues
                logger.debug("Error during ChatStream cleanup: %s", str(e))

                # Still try to close the span even if metrics recording failed
                try:
                    if self._span and self._span.is_recording():
                        self._span.set_status(
                            Status(StatusCode.ERROR, "Cleanup failed"))
                        self._span.end()
                    self._cleanup_completed = True
                except Exception:
                    # Final fallback - just mark as completed to prevent infinite loops
                    self._cleanup_completed = True

    @dont_throw
    def _record_partial_metrics(self):
        """Record metrics based on available partial data"""
        # Always record duration if we have start time
        if self._start_time and isinstance(self._start_time, (float, int)) and self._duration_histogram:
            duration = time.time() - self._start_time
            self._duration_histogram.record(
                duration, attributes=self._shared_attributes()
            )

        # Record basic span attributes even without complete response
        if self._span and self._span.is_recording():
            _set_response_attributes(self._span, self._complete_response)

        # Record partial token metrics if we have any data
        if self._complete_response.get("choices") or self._request_kwargs:
            _set_streaming_token_metrics(
                self._request_kwargs,
                self._complete_response,
                self._span,
                self._token_counter,
                self._shared_attributes(),
            )

        # Record choice metrics if we have any choices processed
        if self._choice_counter and self._complete_response.get("choices"):
            _set_choice_counter_metrics(
                self._choice_counter,
                self._complete_response.get("choices"),
                self._shared_attributes(),
            )


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
    complete_response = {"choices": [], "model": "", "id": ""}

    first_token = True
    time_of_first_token = start_time  # will be updated when first token is received

    for item in response:
        span.add_event(name=f"{SpanAttributes.LLM_CONTENT_COMPLETION_CHUNK}")

        item_to_yield = item

        if first_token and streaming_time_to_first_token:
            time_of_first_token = time.time()
            streaming_time_to_first_token.record(
                time_of_first_token - start_time)
            first_token = False

        _accumulate_stream_items(item, complete_response)

        yield item_to_yield

    shared_attributes = {
        SpanAttributes.LLM_RESPONSE_MODEL: complete_response.get("model") or None,
        "server.address": _get_openai_base_url(instance),
        "stream": True,
    }

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
    if should_emit_events():
        for choice in complete_response.get("choices", []):
            emit_event(_parse_choice_event(choice))
    else:
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
    complete_response = {"choices": [], "model": "", "id": ""}

    first_token = True
    time_of_first_token = start_time  # will be updated when first token is received

    async for item in response:
        span.add_event(name=f"{SpanAttributes.LLM_CONTENT_COMPLETION_CHUNK}")

        item_to_yield = item

        if first_token and streaming_time_to_first_token:
            time_of_first_token = time.time()
            streaming_time_to_first_token.record(
                time_of_first_token - start_time)
            first_token = False

        _accumulate_stream_items(item, complete_response)

        yield item_to_yield

    shared_attributes = {
        SpanAttributes.LLM_RESPONSE_MODEL: complete_response.get("model") or None,
        "server.address": _get_openai_base_url(instance),
        "stream": True,
    }

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
    if should_emit_events():
        for choice in complete_response.get("choices", []):
            emit_event(_parse_choice_event(choice))
    else:
        if should_send_prompts():
            _set_completions(span, complete_response.get("choices"))

    span.set_status(Status(StatusCode.OK))
    span.end()


# pydantic.BaseModel here is ChatCompletionMessageFunctionToolCall (as of openai 1.99.7)
# but we keep to a parent type to support older versions
def _parse_tool_calls(
    tool_calls: Optional[List[Union[dict, pydantic.BaseModel]]],
) -> Union[List[ToolCall], None]:
    """
    Util to correctly parse the tool calls data from the OpenAI API to this module's
    standard `ToolCall`.
    """
    if tool_calls is None:
        return tool_calls

    result = []

    for tool_call in tool_calls:
        tool_call_data = None

        if isinstance(tool_call, dict):
            tool_call_data = copy.deepcopy(tool_call)
        elif _is_chat_message_function_tool_call(tool_call):
            tool_call_data = tool_call.model_dump()
        elif _is_function_call(tool_call):
            function_call = tool_call.model_dump()
            tool_call_data = ToolCall(
                id="",
                function={
                    "name": function_call.get("name"),
                    "arguments": function_call.get("arguments"),
                },
                type="function",
            )

        result.append(tool_call_data)
    return result


def _is_chat_message_function_tool_call(model: Union[dict, pydantic.BaseModel]) -> bool:
    try:
        from openai.types.chat.chat_completion_message_function_tool_call import (
            ChatCompletionMessageFunctionToolCall,
        )

        return isinstance(model, ChatCompletionMessageFunctionToolCall)
    except Exception:
        try:
            # Since OpenAI 1.99.3, ChatCompletionMessageToolCall is a Union,
            # and the isinstance check will fail. This is fine, because in all
            # those versions, the check above will succeed.
            from openai.types.chat.chat_completion_message_tool_call import (
                ChatCompletionMessageToolCall,
            )
            return isinstance(model, ChatCompletionMessageToolCall)
        except Exception:
            return False


def _is_function_call(model: Union[dict, pydantic.BaseModel]) -> bool:
    try:
        from openai.types.chat.chat_completion_message import FunctionCall
        return isinstance(model, FunctionCall)
    except Exception:
        return False


@singledispatch
def _parse_choice_event(choice) -> ChoiceEvent:
    has_message = choice.message is not None
    has_finish_reason = choice.finish_reason is not None
    has_tool_calls = has_message and choice.message.tool_calls
    has_function_call = has_message and choice.message.function_call

    content = choice.message.content if has_message else None
    role = choice.message.role if has_message else "unknown"
    finish_reason = choice.finish_reason if has_finish_reason else "unknown"

    if has_tool_calls and has_function_call:
        tool_calls = choice.message.tool_calls + [choice.message.function_call]
    elif has_tool_calls:
        tool_calls = choice.message.tool_calls
    elif has_function_call:
        tool_calls = [choice.message.function_call]
    else:
        tool_calls = None

    return ChoiceEvent(
        index=choice.index,
        message={"content": content, "role": role},
        finish_reason=finish_reason,
        tool_calls=_parse_tool_calls(tool_calls),
    )


@_parse_choice_event.register
def _(choice: dict) -> ChoiceEvent:
    message = choice.get("message")
    has_message = message is not None
    has_finish_reason = choice.get("finish_reason") is not None
    has_tool_calls = has_message and message.get("tool_calls")
    has_function_call = has_message and message.get("function_call")

    content = choice.get("message").get("content", "") if has_message else None
    role = choice.get("message").get("role") if has_message else "unknown"
    finish_reason = choice.get(
        "finish_reason") if has_finish_reason else "unknown"

    if has_tool_calls and has_function_call:
        tool_calls = message.get("tool_calls") + [message.get("function_call")]
    elif has_tool_calls:
        tool_calls = message.get("tool_calls")
    elif has_function_call:
        tool_calls = [message.get("function_call")]
    else:
        tool_calls = None

    if tool_calls is not None:
        for tool_call in tool_calls:
            tool_call["type"] = "function"

    return ChoiceEvent(
        index=choice.get("index"),
        message={"content": content, "role": role},
        finish_reason=finish_reason,
        tool_calls=tool_calls,
    )


def _accumulate_stream_items(item, complete_response):
    if is_openai_v1():
        item = model_as_dict(item)

    complete_response["model"] = item.get("model")
    complete_response["id"] = item.get("id")

    # capture usage information from the last stream chunks
    if item.get("usage"):
        complete_response["usage"] = item.get("usage")
    elif item.get("choices") and item["choices"][0].get("usage"):
        # Some LLM providers like moonshot mistakenly place token usage information within choices[0], handle this.
        complete_response["usage"] = item["choices"][0].get("usage")

    # prompt filter results
    if item.get("prompt_filter_results"):
        complete_response["prompt_filter_results"] = item.get(
            "prompt_filter_results")

    for choice in item.get("choices"):
        index = choice.get("index")
        if len(complete_response.get("choices")) <= index:
            complete_response["choices"].append(
                {"index": index, "message": {"content": "", "role": ""}}
            )
        complete_choice = complete_response.get("choices")[index]
        if choice.get("finish_reason"):
            complete_choice["finish_reason"] = choice.get("finish_reason")
        if choice.get("content_filter_results"):
            complete_choice["content_filter_results"] = choice.get(
                "content_filter_results"
            )

        delta = choice.get("delta")

        if delta and delta.get("content"):
            complete_choice["message"]["content"] += delta.get("content")

        if delta and delta.get("role"):
            complete_choice["message"]["role"] = delta.get("role")
        if delta and delta.get("tool_calls"):
            tool_calls = delta.get("tool_calls")
            if not isinstance(tool_calls, list) or len(tool_calls) == 0:
                continue

            if not complete_choice["message"].get("tool_calls"):
                complete_choice["message"]["tool_calls"] = []

            for tool_call in tool_calls:
                i = int(tool_call["index"])
                if len(complete_choice["message"]["tool_calls"]) <= i:
                    complete_choice["message"]["tool_calls"].append(
                        {"id": "", "function": {"name": "", "arguments": ""}}
                    )

                span_tool_call = complete_choice["message"]["tool_calls"][i]
                span_function = span_tool_call["function"]
                tool_call_function = tool_call.get("function")

                if tool_call.get("id"):
                    span_tool_call["id"] = tool_call.get("id")
                if tool_call_function and tool_call_function.get("name"):
                    span_function["name"] = tool_call_function.get("name")
                if tool_call_function and tool_call_function.get("arguments"):
                    span_function["arguments"] += tool_call_function.get(
                        "arguments")
