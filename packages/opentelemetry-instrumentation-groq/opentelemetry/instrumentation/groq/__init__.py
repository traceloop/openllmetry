"""OpenTelemetry Groq instrumentation"""

import logging
import os
import time
from typing import Callable, Collection, Union

from opentelemetry import context as context_api
from opentelemetry._events import EventLogger, get_event_logger
from opentelemetry.instrumentation.groq.config import Config
from opentelemetry.instrumentation.groq.event_emitter import (
    emit_choice_events,
    emit_message_events,
    emit_streaming_response_events,
)
from opentelemetry.instrumentation.groq.span_utils import (
    set_input_attributes,
    set_model_input_attributes,
    set_model_response_attributes,
    set_model_streaming_response_attributes,
    set_response_attributes,
    set_streaming_response_attributes,
)
from opentelemetry.instrumentation.groq.utils import (
    error_metrics_attributes,
    shared_metrics_attributes,
    should_emit_events,
)
from opentelemetry.instrumentation.groq.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap
from opentelemetry.metrics import Counter, Histogram, Meter, get_meter
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    Meters,
    SpanAttributes,
)
from opentelemetry.trace import SpanKind, Tracer, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

from groq._streaming import AsyncStream, Stream

logger = logging.getLogger(__name__)

_instruments = ("groq >= 0.9.0",)


WRAPPED_METHODS = [
    {
        "package": "groq.resources.chat.completions",
        "object": "Completions",
        "method": "create",
        "span_name": "groq.chat",
    },
]
WRAPPED_AMETHODS = [
    {
        "package": "groq.resources.chat.completions",
        "object": "AsyncCompletions",
        "method": "create",
        "span_name": "groq.chat",
    },
]


def is_streaming_response(response):
    return isinstance(response, Stream) or isinstance(response, AsyncStream)


def _with_chat_telemetry_wrapper(func):
    """Helper for providing tracer for wrapper functions. Includes metric collectors."""

    def _with_chat_telemetry(
        tracer,
        token_histogram,
        choice_counter,
        duration_histogram,
        event_logger,
        to_wrap,
    ):
        def wrapper(wrapped, instance, args, kwargs):
            return func(
                tracer,
                token_histogram,
                choice_counter,
                duration_histogram,
                event_logger,
                to_wrap,
                wrapped,
                instance,
                args,
                kwargs,
            )

        return wrapper

    return _with_chat_telemetry


def _create_metrics(meter: Meter):
    token_histogram = meter.create_histogram(
        name=Meters.LLM_TOKEN_USAGE,
        unit="token",
        description="Measures number of input and output tokens used",
    )

    choice_counter = meter.create_counter(
        name=Meters.LLM_GENERATION_CHOICES,
        unit="choice",
        description="Number of choices returned by chat completions call",
    )

    duration_histogram = meter.create_histogram(
        name=Meters.LLM_OPERATION_DURATION,
        unit="s",
        description="GenAI operation duration",
    )

    return token_histogram, choice_counter, duration_histogram


def _process_streaming_chunk(chunk):
    """Extract content, finish_reason and usage from a streaming chunk."""
    if not chunk.choices:
        return None, None, None

    delta = chunk.choices[0].delta
    content = delta.content if hasattr(delta, "content") else None
    finish_reason = chunk.choices[0].finish_reason

    # Extract usage from x_groq if present in the final chunk
    usage = None
    if hasattr(chunk, "x_groq") and chunk.x_groq and chunk.x_groq.usage:
        usage = chunk.x_groq.usage

    return content, finish_reason, usage


def _handle_streaming_response(
    span, accumulated_content, finish_reason, usage, event_logger
):
    set_model_streaming_response_attributes(span, usage)
    if should_emit_events() and event_logger:
        emit_streaming_response_events(accumulated_content, finish_reason, event_logger)
    else:
        set_streaming_response_attributes(
            span, accumulated_content, finish_reason, usage
        )


def _create_stream_processor(response, span, event_logger):
    """Create a generator that processes a stream while collecting telemetry."""
    accumulated_content = ""
    finish_reason = None
    usage = None

    for chunk in response:
        content, chunk_finish_reason, chunk_usage = _process_streaming_chunk(chunk)
        if content:
            accumulated_content += content
        if chunk_finish_reason:
            finish_reason = chunk_finish_reason
        if chunk_usage:
            usage = chunk_usage
        yield chunk

    _handle_streaming_response(
        span, accumulated_content, finish_reason, usage, event_logger
    )

    if span.is_recording():
        span.set_status(Status(StatusCode.OK))

    span.end()


async def _create_async_stream_processor(response, span, event_logger):
    """Create an async generator that processes a stream while collecting telemetry."""
    accumulated_content = ""
    finish_reason = None
    usage = None

    async for chunk in response:
        content, chunk_finish_reason, chunk_usage = _process_streaming_chunk(chunk)
        if content:
            accumulated_content += content
        if chunk_finish_reason:
            finish_reason = chunk_finish_reason
        if chunk_usage:
            usage = chunk_usage
        yield chunk

    _handle_streaming_response(
        span, accumulated_content, finish_reason, usage, event_logger
    )

    if span.is_recording():
        span.set_status(Status(StatusCode.OK))

    span.end()


def _handle_input(span, kwargs, event_logger):
    set_model_input_attributes(span, kwargs)
    if should_emit_events() and event_logger:
        emit_message_events(kwargs, event_logger)
    else:
        set_input_attributes(span, kwargs)


def _handle_response(span, response, token_histogram, event_logger):
    set_model_response_attributes(span, response, token_histogram)
    if should_emit_events() and event_logger:
        emit_choice_events(response, event_logger)
    else:
        set_response_attributes(span, response)


@_with_chat_telemetry_wrapper
def _wrap(
    tracer: Tracer,
    token_histogram: Histogram,
    choice_counter: Counter,
    duration_histogram: Histogram,
    event_logger: Union[EventLogger, None],
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
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Groq",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
    )

    _handle_input(span, kwargs, event_logger)

    start_time = time.time()
    try:
        response = wrapped(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        end_time = time.time()
        attributes = error_metrics_attributes(e)

        if duration_histogram:
            duration = end_time - start_time
            duration_histogram.record(duration, attributes=attributes)

        raise e

    end_time = time.time()

    if is_streaming_response(response):
        try:
            return _create_stream_processor(response, span, event_logger)
        except Exception as ex:
            logger.warning(
                "Failed to process streaming response for groq span, error: %s",
                str(ex),
            )
            span.set_status(Status(StatusCode.ERROR))
            span.end()
            raise
    elif response:
        try:
            metric_attributes = shared_metrics_attributes(response)

            if duration_histogram:
                duration = time.time() - start_time
                duration_histogram.record(
                    duration,
                    attributes=metric_attributes,
                )

            _handle_response(span, response, token_histogram, event_logger)

        except Exception as ex:  # pylint: disable=broad-except
            logger.warning(
                "Failed to set response attributes for groq span, error: %s",
                str(ex),
            )

        if span.is_recording():
            span.set_status(Status(StatusCode.OK))
    span.end()
    return response


@_with_chat_telemetry_wrapper
async def _awrap(
    tracer,
    token_histogram: Histogram,
    choice_counter: Counter,
    duration_histogram: Histogram,
    event_logger: Union[EventLogger, None],
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
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Groq",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
    )

    _handle_input(span, kwargs, event_logger)

    start_time = time.time()

    try:
        response = await wrapped(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        end_time = time.time()
        attributes = error_metrics_attributes(e)

        if duration_histogram:
            duration = end_time - start_time
            duration_histogram.record(duration, attributes=attributes)

        raise e

    end_time = time.time()

    if is_streaming_response(response):
        try:
            return await _create_async_stream_processor(response, span, event_logger)
        except Exception as ex:
            logger.warning(
                "Failed to process streaming response for groq span, error: %s",
                str(ex),
            )
            span.set_status(Status(StatusCode.ERROR))
            span.end()
            raise
    elif response:
        metric_attributes = shared_metrics_attributes(response)

        if duration_histogram:
            duration = time.time() - start_time
            duration_histogram.record(
                duration,
                attributes=metric_attributes,
            )

        _handle_response(span, response, token_histogram, event_logger)

        if span.is_recording():
            span.set_status(Status(StatusCode.OK))
    span.end()
    return response


def is_metrics_enabled() -> bool:
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"


class GroqInstrumentor(BaseInstrumentor):
    """An instrumentor for Groq's client library."""

    def __init__(
        self,
        enrich_token_usage: bool = False,
        exception_logger=None,
        use_legacy_attributes: bool = True,
        get_common_metrics_attributes: Callable[[], dict] = lambda: {},
    ):
        super().__init__()
        Config.exception_logger = exception_logger
        Config.enrich_token_usage = enrich_token_usage
        Config.get_common_metrics_attributes = get_common_metrics_attributes
        Config.use_legacy_attributes = use_legacy_attributes

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        # meter and counters are inited here
        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        if is_metrics_enabled():
            (
                token_histogram,
                choice_counter,
                duration_histogram,
            ) = _create_metrics(meter)
        else:
            (
                token_histogram,
                choice_counter,
                duration_histogram,
            ) = (None, None, None)

        event_logger = None
        if not Config.use_legacy_attributes:
            event_logger_provider = kwargs.get("event_logger_provider")
            event_logger = get_event_logger(
                __name__, __version__, event_logger_provider=event_logger_provider
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
                        choice_counter,
                        duration_histogram,
                        event_logger,
                        wrapped_method,
                    ),
                )
            except ModuleNotFoundError:
                pass  # that's ok, we don't want to fail if some methods do not exist

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
                        choice_counter,
                        duration_histogram,
                        event_logger,
                        wrapped_method,
                    ),
                )
            except ModuleNotFoundError:
                pass  # that's ok, we don't want to fail if some methods do not exist

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
