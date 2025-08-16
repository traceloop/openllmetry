"""OpenTelemetry Writer instrumentation"""

import logging
import os
import time
from typing import Collection, Union

from opentelemetry._events import EventLogger, get_event_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap
from opentelemetry.metrics import Counter, Histogram, Meter, get_meter
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics as GenAIMetrics
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    Meters,
    SpanAttributes,
)
from opentelemetry.trace import SpanKind, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper
from writerai._streaming import AsyncStream, Stream

from opentelemetry import context as context_api
from opentelemetry.instrumentation.writer.config import Config
from opentelemetry.instrumentation.writer.span_utils import (
    set_input_attributes,
    set_model_input_attributes,
)
from opentelemetry.instrumentation.writer.utils import (
    error_metrics_attributes,
    response_attributes,
    should_emit_events,
)
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
    return isinstance(response, Stream) or isinstance(response, AsyncStream)


def _handle_input(span, kwargs, event_logger):
    set_model_input_attributes(span, kwargs)
    if should_emit_events() and event_logger:
        ...  # TODO message events emitter
    else:
        set_input_attributes(span, kwargs)


def _request_type_by_method(method_name):
    if method_name == "chat":
        return LLMRequestTypeValues.CHAT
    elif method_name == "create":
        return LLMRequestTypeValues.COMPLETION
    else:
        return LLMRequestTypeValues.UNKNOWN


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions. Includes metric collectors."""

    def _with_chat_telemetry(
        tracer,
        token_histogram,
        duration_histogram,
        streaming_time_to_first_token,
        streaming_time_to_generate,
        choice_counter,
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
                choice_counter,
                event_logger,
                to_wrap,
                wrapped,
                instance,
                args,
                kwargs,
            )

        return wrapper

    return _with_chat_telemetry


@_with_tracer_wrapper
def _wrap(
    tracer,
    token_histogram: Histogram,
    duration_histogram: Histogram,
    streaming_time_to_first_token: Histogram,
    streaming_time_to_generate: Histogram,
    choice_counter: Counter,
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
    request_type = _request_type_by_method(to_wrap.get("method"))

    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Writer",
            SpanAttributes.LLM_REQUEST_TYPE: request_type.value,
        },
    )

    _handle_input(span, kwargs, event_logger)

    start_time = time.time()
    try:
        response = wrapped(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        end_time = time.time()

        if duration_histogram:
            duration = end_time - start_time
            duration_histogram.record(duration, attributes=error_metrics_attributes(e))

        raise e

    if is_streaming_response(response):
        try:
            ...  # TODO streaming response processor method
        except Exception as ex:
            logger.warning(
                "Failed to process streaming response for writer span, error: %s",
                str(ex),
            )
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
                    attributes=response_attributes(response),
                )

            # TODO non-streaming response processor method

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
    choice_counter: Counter,
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
    request_type = _request_type_by_method(to_wrap.get("method"))

    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Writer",
            SpanAttributes.LLM_REQUEST_TYPE: request_type.value,
        },
    )

    _handle_input(span, kwargs, event_logger)

    start_time = time.time()
    try:
        response = await wrapped(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        end_time = time.time()

        if duration_histogram:
            duration = end_time - start_time
            duration_histogram.record(duration, attributes=error_metrics_attributes(e))

        raise e

    if is_streaming_response(response):
        try:
            ...  # TODO async streaming response processor method
        except Exception as ex:
            logger.warning(
                "Failed to process streaming response for writer span, error: %s",
                str(ex),
            )
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
                    attributes=response_attributes(response),
                )

            # TODO non-streaming response processor method

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

    choice_counter = meter.create_counter(
        name=Meters.LLM_GENERATION_CHOICES,
        unit="choice",
        description="Number of choices returned by chat completions call",
    )

    return (
        token_histogram,
        duration_histogram,
        streaming_time_to_first_token,
        streaming_time_to_generate,
        choice_counter,
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
                choice_counter,
            ) = _build_metrics(meter)
        else:
            (
                token_histogram,
                duration_histogram,
                streaming_time_to_first_token,
                streaming_time_to_generate,
                choice_counter,
            ) = (None, None, None, None, None)

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
                        duration_histogram,
                        streaming_time_to_first_token,
                        streaming_time_to_generate,
                        choice_counter,
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
                        choice_counter,
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
