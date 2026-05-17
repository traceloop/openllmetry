"""OpenTelemetry Writer instrumentation"""

import logging
import os
import time
from typing import Collection, Union

from opentelemetry._logs import Logger, get_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (_SUPPRESS_INSTRUMENTATION_KEY,
                                                 unwrap)
from opentelemetry.metrics import Histogram, Meter, get_meter
from opentelemetry.semconv._incubating.metrics import \
    gen_ai_metrics as GenAIMetrics
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, Meters, SpanAttributes)
from opentelemetry.trace import SpanKind, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper
from writerai._streaming import AsyncStream, Stream
from writerai.types import (ChatCompletion, ChatCompletionChunk, Completion,
                            CompletionChunk)
from writerai.types.completion import Choice

from opentelemetry import context as context_api
from opentelemetry.instrumentation.writer.config import Config
from opentelemetry.instrumentation.writer.event_emitter import (
    emit_choice_events, emit_message_events)
from opentelemetry.instrumentation.writer.span_utils import (
    set_input_attributes, set_model_input_attributes,
    set_model_response_attributes, set_response_attributes)
from opentelemetry.instrumentation.writer.utils import (
    enhance_list_size, error_metrics_attributes,
    initialize_accumulated_response, initialize_choice, initialize_tool_call,
    request_type_by_method, response_attributes, should_emit_events)
from opentelemetry.instrumentation.writer.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("writer-sdk >= 2.2.1, < 3",)

WRAPPED_METHODS = [
    {
        "package": "writerai.resources.chat",
        "object": "ChatResource",
        "method": "chat",
        "span_name": "writerai.chat",
    },
    {
        "package": "writerai.resources.completions",
        "object": "CompletionsResource",
        "method": "create",
        "span_name": "writerai.completions",
    },
]
WRAPPED_AMETHODS = [
    {
        "package": "writerai.resources.chat",
        "object": "AsyncChatResource",
        "method": "chat",
        "span_name": "writerai.chat",
    },
    {
        "package": "writerai.resources.completions",
        "object": "AsyncCompletionsResource",
        "method": "create",
        "span_name": "writerai.completions",
    },
]


def is_streaming_response(response):
    return isinstance(response, (Stream, AsyncStream))


def _update_accumulated_response(accumulated_response, chunk):
    if isinstance(accumulated_response, ChatCompletion) and isinstance(
        chunk, ChatCompletionChunk
    ):
        if chunk.service_tier:
            accumulated_response.service_tier = chunk.service_tier
        if chunk.system_fingerprint:
            accumulated_response.system_fingerprint = chunk.system_fingerprint
        if chunk.model:
            accumulated_response.model = chunk.model
        if chunk.usage:
            accumulated_response.usage = chunk.usage
        if chunk.created:
            accumulated_response.created = chunk.created

        if chunk.choices:
            choice_index = chunk.choices[0].index or 0

            try:
                accumulated_response.choices[choice_index]
            except IndexError:
                enhance_list_size(accumulated_response.choices, choice_index + 1)
                accumulated_response.choices[choice_index] = initialize_choice()

            if finish_reason := chunk.choices[0].finish_reason:
                accumulated_response.choices[choice_index].finish_reason = finish_reason
            accumulated_response.choices[choice_index].index = choice_index

            if chunk.choices[0].delta:
                if content := chunk.choices[0].delta.content:
                    accumulated_response.choices[
                        choice_index
                    ].message.content += content
                if role := chunk.choices[0].delta.role:
                    accumulated_response.choices[choice_index].message.role = role
                if chunk.choices[0].delta.tool_calls:
                    tool_index = chunk.choices[0].delta.tool_calls[0].index or 0

                    try:
                        accumulated_response.choices[choice_index].message.tool_calls[
                            tool_index
                        ]
                    except IndexError:
                        enhance_list_size(
                            accumulated_response.choices[
                                choice_index
                            ].message.tool_calls,
                            tool_index + 1,
                        )
                        accumulated_response.choices[choice_index].message.tool_calls[
                            tool_index
                        ] = initialize_tool_call()

                    if name := chunk.choices[0].delta.tool_calls[0].function.name:
                        accumulated_response.choices[choice_index].message.tool_calls[
                            tool_index
                        ].function.name += name
                    if (
                        arguments := chunk.choices[0]
                        .delta.tool_calls[0]
                        .function.arguments
                    ):
                        accumulated_response.choices[choice_index].message.tool_calls[
                            tool_index
                        ].function.arguments += arguments
                    if tool_id := chunk.choices[0].delta.tool_calls[0].id:
                        accumulated_response.choices[choice_index].message.tool_calls[
                            tool_index
                        ].id = tool_id

                    accumulated_response.choices[choice_index].message.tool_calls[
                        tool_index
                    ].index = tool_index

    elif isinstance(accumulated_response, Completion) and isinstance(
        chunk, CompletionChunk
    ):
        if chunk.value:
            if accumulated_response.choices and accumulated_response.choices[0].text:
                accumulated_response.choices[0].text += chunk.value
            else:
                accumulated_response.choices = [Choice(text=chunk.value)]
    else:
        raise ValueError(
            f"Accumulated response and chunk types mismatch: {type(accumulated_response)}, {type(chunk)}"
        )


def _create_stream_processor(
    response,
    span,
    event_logger,
    start_time,
    duration_histogram,
    streaming_time_to_first_token,
    streaming_time_to_generate,
    token_histogram,
    method,
):
    accumulated_response = initialize_accumulated_response(response)
    first_token_time = None
    last_token_time = start_time
    error: Exception | None = None

    try:
        for chunk in response:
            if first_token_time is None:
                first_token_time = time.time()

            _update_accumulated_response(accumulated_response, chunk)

            yield chunk

        last_token_time = time.time()
    except Exception as ex:
        error = ex
        if span.is_recording():
            span.set_status(Status(StatusCode.ERROR))
        raise
    finally:
        metrics_attributes = response_attributes(accumulated_response, method)
        metrics_attributes.update({"stream": True})

        if streaming_time_to_first_token:
            ttft = (first_token_time or last_token_time) - start_time
            streaming_time_to_first_token.record(ttft, attributes=metrics_attributes)

        if streaming_time_to_generate:
            streaming_time_to_generate.record(
                last_token_time - (first_token_time or last_token_time),
                attributes=metrics_attributes,
            )

        if duration_histogram:
            duration_histogram.record(
                last_token_time - start_time, attributes=metrics_attributes
            )

        _handle_response(
            span, accumulated_response, token_histogram, event_logger, method
        )

        if span.is_recording() and error is None:
            span.set_status(Status(StatusCode.OK))

        span.end()


async def _create_async_stream_processor(
    response,
    span,
    event_logger,
    start_time,
    duration_histogram,
    streaming_time_to_first_token,
    streaming_time_to_generate,
    token_histogram,
    method,
):
    accumulated_response = initialize_accumulated_response(response)
    first_token_time = None
    last_token_time = start_time
    error: Exception | None = None

    try:
        async for chunk in response:
            if first_token_time is None:
                first_token_time = time.time()

            _update_accumulated_response(accumulated_response, chunk)

            yield chunk

        last_token_time = time.time()
    except Exception as ex:
        error = ex
        if span.is_recording():
            span.set_status(Status(StatusCode.ERROR))
        raise
    finally:
        metrics_attributes = response_attributes(accumulated_response, method)
        metrics_attributes.update({"stream": True})

        if streaming_time_to_first_token:
            ttft = (first_token_time or last_token_time) - start_time
            streaming_time_to_first_token.record(ttft, attributes=metrics_attributes)

        if streaming_time_to_generate:
            streaming_time_to_generate.record(
                last_token_time - (first_token_time or last_token_time),
                attributes=metrics_attributes,
            )

        if duration_histogram:
            duration_histogram.record(
                last_token_time - start_time, attributes=metrics_attributes
            )

        _handle_response(
            span, accumulated_response, token_histogram, event_logger, method
        )

        if span.is_recording() and error is None:
            span.set_status(Status(StatusCode.OK))

        span.end()


def _handle_input(span, kwargs, event_logger):
    set_model_input_attributes(span, kwargs)
    if should_emit_events() and event_logger:
        emit_message_events(kwargs, event_logger)
    else:
        set_input_attributes(span, kwargs)


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions. Includes metric collectors."""

    def _with_chat_telemetry(
        tracer,
        token_histogram,
        duration_histogram,
        streaming_time_to_first_token,
        streaming_time_to_generate,
        event_logger,
        to_wrap,
    ):
        def wrapper(wrapped, instance, args, kwargs):
            return func(
                tracer,
                token_histogram,
                duration_histogram,
                streaming_time_to_first_token,
                streaming_time_to_generate,
                event_logger,
                to_wrap,
                wrapped,
                instance,
                args,
                kwargs,
            )

        return wrapper

    return _with_chat_telemetry


def _handle_response(span, response, token_histogram, event_logger, method):
    set_model_response_attributes(span, response, token_histogram, method)
    if should_emit_events() and event_logger:
        emit_choice_events(response, event_logger)
    else:
        set_response_attributes(span, response)


@_with_tracer_wrapper
def _wrap(
    tracer,
    token_histogram: Histogram,
    duration_histogram: Histogram,
    streaming_time_to_first_token: Histogram,
    streaming_time_to_generate: Histogram,
    event_logger: Union[Logger, None],
    to_wrap,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    request_type = request_type_by_method(to_wrap.get("method"))

    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "writer",
            SpanAttributes.LLM_REQUEST_TYPE: request_type.value,
        },
    )

    _handle_input(span, kwargs, event_logger)

    start_time = time.time()
    try:
        response = wrapped(*args, **kwargs)
    except Exception as ex:  # pylint: disable=broad-except
        end_time = time.time()

        if duration_histogram:
            duration = end_time - start_time
            duration_histogram.record(duration, attributes=error_metrics_attributes(ex))

        if span.is_recording():
            span.set_status(Status(StatusCode.ERROR))

        span.end()
        raise

    if is_streaming_response(response):
        try:
            return _create_stream_processor(
                response,
                span,
                event_logger,
                start_time,
                duration_histogram,
                streaming_time_to_first_token,
                streaming_time_to_generate,
                token_histogram,
                to_wrap.get("method"),
            )
        except Exception as ex:
            logger.warning(
                "Failed to process streaming response for writer span, error: %s",
                str(ex),
            )
            if span.is_recording():
                span.set_status(Status(StatusCode.ERROR))

            span.end()
            raise

    elif response:
        end_time = time.time()
        try:
            if duration_histogram:
                duration = end_time - start_time
                duration_histogram.record(
                    duration,
                    attributes=response_attributes(response, to_wrap.get("method")),
                )

            _handle_response(
                span, response, token_histogram, event_logger, to_wrap.get("method")
            )

        except Exception as ex:  # pylint: disable=broad-except
            logger.warning(
                "Failed to set response attributes for writer span, error: %s",
                str(ex),
            )

        if span.is_recording():
            span.set_status(Status(StatusCode.OK))

    span.end()
    return response


@_with_tracer_wrapper
async def _awrap(
    tracer,
    token_histogram: Histogram,
    duration_histogram: Histogram,
    streaming_time_to_first_token: Histogram,
    streaming_time_to_generate: Histogram,
    event_logger: Union[Logger, None],
    to_wrap,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return await wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    request_type = request_type_by_method(to_wrap.get("method"))

    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "writer",
            SpanAttributes.LLM_REQUEST_TYPE: request_type.value,
        },
    )

    _handle_input(span, kwargs, event_logger)

    start_time = time.time()
    try:
        response = await wrapped(*args, **kwargs)
    except Exception as ex:  # pylint: disable=broad-except
        end_time = time.time()

        if duration_histogram:
            duration = end_time - start_time
            duration_histogram.record(duration, attributes=error_metrics_attributes(ex))

        if span.is_recording():
            span.set_status(Status(StatusCode.ERROR))

        span.end()
        raise

    if is_streaming_response(response):
        try:
            return _create_async_stream_processor(
                response,
                span,
                event_logger,
                start_time,
                duration_histogram,
                streaming_time_to_first_token,
                streaming_time_to_generate,
                token_histogram,
                to_wrap.get("method"),
            )
        except Exception as ex:
            logger.warning(
                "Failed to process streaming response for writer span, error: %s",
                str(ex),
            )
            if span.is_recording():
                span.set_status(Status(StatusCode.ERROR))

            span.end()
            raise

    elif response:
        end_time = time.time()
        try:
            if duration_histogram:
                duration = end_time - start_time
                duration_histogram.record(
                    duration,
                    attributes=response_attributes(response, to_wrap.get("method")),
                )

            _handle_response(
                span, response, token_histogram, event_logger, to_wrap.get("method")
            )

        except Exception as ex:  # pylint: disable=broad-except
            logger.warning(
                "Failed to set response attributes for writer span, error: %s",
                str(ex),
            )

        if span.is_recording():
            span.set_status(Status(StatusCode.OK))

    span.end()
    return response


def _build_metrics(meter: Meter):
    token_histogram = meter.create_histogram(
        name=Meters.LLM_TOKEN_USAGE,
        unit="token",
        description="Measures number of input and output tokens used",
    )

    duration_histogram = meter.create_histogram(
        name=Meters.LLM_OPERATION_DURATION,
        unit="s",
        description="Generation operation duration",
    )

    streaming_time_to_first_token = meter.create_histogram(
        name=GenAIMetrics.GEN_AI_SERVER_TIME_TO_FIRST_TOKEN,
        unit="s",
        description="Time to first token in streaming chat completions",
    )

    streaming_time_to_generate = meter.create_histogram(
        name=Meters.LLM_STREAMING_TIME_TO_GENERATE,
        unit="s",
        description="Time from first token to completion in streaming responses",
    )

    return (
        token_histogram,
        duration_histogram,
        streaming_time_to_first_token,
        streaming_time_to_generate,
    )


def is_metrics_collection_enabled() -> bool:
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"


class WriterInstrumentor(BaseInstrumentor):
    """An instrumentor for Writer's client library."""

    def __init__(self, exception_logger=None, use_legacy_attributes=True):
        super().__init__()
        Config.exception_logger = exception_logger
        Config.use_legacy_attributes = use_legacy_attributes

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        if is_metrics_collection_enabled():
            (
                token_histogram,
                duration_histogram,
                streaming_time_to_first_token,
                streaming_time_to_generate,
            ) = _build_metrics(meter)
        else:
            (
                token_histogram,
                duration_histogram,
                streaming_time_to_first_token,
                streaming_time_to_generate,
            ) = (None, None, None, None)

        event_logger = None
        if not Config.use_legacy_attributes:
            logger_provider = kwargs.get("logger_provider")
            event_logger = get_logger(
                __name__, __version__, logger_provider=logger_provider
            )

        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            try:
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}",
                    _wrap(
                        tracer,
                        token_histogram,
                        duration_histogram,
                        streaming_time_to_first_token,
                        streaming_time_to_generate,
                        event_logger,
                        wrapped_method,
                    ),
                )
            except ModuleNotFoundError:
                pass

        for wrapped_method in WRAPPED_AMETHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            try:
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}",
                    _awrap(
                        tracer,
                        token_histogram,
                        duration_histogram,
                        streaming_time_to_first_token,
                        streaming_time_to_generate,
                        event_logger,
                        wrapped_method,
                    ),
                )
            except ModuleNotFoundError:
                pass

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"{wrap_package}.{wrap_object}",
                wrapped_method.get("method"),
            )
        for wrapped_method in WRAPPED_AMETHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"{wrap_package}.{wrap_object}",
                wrapped_method.get("method"),
            )
