"""OpenTelemetry Groq instrumentation"""

import json
import logging
import os
import time
from typing import Callable, Collection

from groq._streaming import AsyncStream, Stream
from opentelemetry import context as context_api
from opentelemetry.instrumentation.groq.config import Config
from opentelemetry.instrumentation.groq.utils import (
    dont_throw,
    error_metrics_attributes,
    model_as_dict,
    set_span_attribute,
    shared_metrics_attributes,
    should_send_prompts,
)
from opentelemetry.instrumentation.groq.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap
from opentelemetry.metrics import Counter, Histogram, Meter, get_meter
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    SpanAttributes,
    Meters,
)
from opentelemetry.trace import SpanKind, Tracer, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("groq >= 0.9.0",)

CONTENT_FILTER_KEY = "content_filter_results"

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


def _dump_content(content):
    if isinstance(content, str):
        return content
    json_serializable = []
    for item in content:
        if item.get("type") == "text":
            json_serializable.append({"type": "text", "text": item.get("text")})
        elif item.get("type") == "image":
            json_serializable.append(
                {
                    "type": "image",
                    "source": {
                        "type": item.get("source").get("type"),
                        "media_type": item.get("source").get("media_type"),
                        "data": str(item.get("source").get("data")),
                    },
                }
            )
    return json.dumps(json_serializable)


@dont_throw
def _set_input_attributes(span, kwargs):
    set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
    set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_tokens_to_sample")
    )
    set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TEMPERATURE, kwargs.get("temperature")
    )
    set_span_attribute(span, SpanAttributes.LLM_REQUEST_TOP_P, kwargs.get("top_p"))
    set_span_attribute(
        span, SpanAttributes.LLM_FREQUENCY_PENALTY, kwargs.get("frequency_penalty")
    )
    set_span_attribute(
        span, SpanAttributes.LLM_PRESENCE_PENALTY, kwargs.get("presence_penalty")
    )
    set_span_attribute(span, SpanAttributes.LLM_IS_STREAMING, kwargs.get("stream") or False)

    if should_send_prompts():
        if kwargs.get("prompt") is not None:
            set_span_attribute(
                span, f"{SpanAttributes.LLM_PROMPTS}.0.user", kwargs.get("prompt")
            )

        elif kwargs.get("messages") is not None:
            for i, message in enumerate(kwargs.get("messages")):
                set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                    _dump_content(message.get("content")),
                )
                set_span_attribute(
                    span, f"{SpanAttributes.LLM_PROMPTS}.{i}.role", message.get("role")
                )


def _set_completions(span, choices):
    if choices is None:
        return

    for choice in choices:
        index = choice.get("index")
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
        set_span_attribute(
            span, f"{prefix}.finish_reason", choice.get("finish_reason")
        )

        if choice.get("content_filter_results"):
            set_span_attribute(
                span,
                f"{prefix}.{CONTENT_FILTER_KEY}",
                json.dumps(choice.get("content_filter_results")),
            )

        if choice.get("finish_reason") == "content_filter":
            set_span_attribute(span, f"{prefix}.role", "assistant")
            set_span_attribute(span, f"{prefix}.content", "FILTERED")

            return

        message = choice.get("message")
        if not message:
            return

        set_span_attribute(span, f"{prefix}.role", message.get("role"))
        set_span_attribute(span, f"{prefix}.content", message.get("content"))

        function_call = message.get("function_call")
        if function_call:
            set_span_attribute(
                span, f"{prefix}.tool_calls.0.name", function_call.get("name")
            )
            set_span_attribute(
                span,
                f"{prefix}.tool_calls.0.arguments",
                function_call.get("arguments"),
            )

        tool_calls = message.get("tool_calls")
        if tool_calls:
            for i, tool_call in enumerate(tool_calls):
                function = tool_call.get("function")
                set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.id",
                    tool_call.get("id"),
                )
                set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.name",
                    function.get("name"),
                )
                set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.arguments",
                    function.get("arguments"),
                )


@dont_throw
def _set_response_attributes(span, response):
    response = model_as_dict(response)

    set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, response.get("model"))

    usage = response.get("usage")
    if usage:
        set_span_attribute(
            span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.get("total_tokens")
        )
        set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            usage.get("completion_tokens"),
        )
        set_span_attribute(
            span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.get("prompt_tokens")
        )

    choices = response.get("choices")
    if should_send_prompts() and choices:
        _set_completions(span, choices)


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def _with_chat_telemetry_wrapper(func):
    """Helper for providing tracer for wrapper functions. Includes metric collectors."""

    def _with_chat_telemetry(
        tracer,
        token_histogram,
        choice_counter,
        duration_histogram,
        to_wrap,
    ):
        def wrapper(wrapped, instance, args, kwargs):
            return func(
                tracer,
                token_histogram,
                choice_counter,
                duration_histogram,
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


@_with_chat_telemetry_wrapper
def _wrap(
    tracer: Tracer,
    token_histogram: Histogram,
    choice_counter: Counter,
    duration_histogram: Histogram,
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

    if span.is_recording():
        _set_input_attributes(span, kwargs)

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
        # TODO: implement streaming
        pass
    elif response:
        try:
            metric_attributes = shared_metrics_attributes(response)

            if duration_histogram:
                duration = time.time() - start_time
                duration_histogram.record(
                    duration,
                    attributes=metric_attributes,
                )

            if span.is_recording():
                _set_response_attributes(span, response)

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
    try:
        if span.is_recording():
            _set_input_attributes(span, kwargs)

    except Exception as ex:  # pylint: disable=broad-except
        logger.warning(
            "Failed to set input attributes for groq span, error: %s", str(ex)
        )

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

    if is_streaming_response(response):
        # TODO: implement streaming
        pass
    elif response:
        metric_attributes = shared_metrics_attributes(response)

        if duration_histogram:
            duration = time.time() - start_time
            duration_histogram.record(
                duration,
                attributes=metric_attributes,
            )

        if span.is_recording():
            _set_response_attributes(span, response)

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
        get_common_metrics_attributes: Callable[[], dict] = lambda: {},
    ):
        super().__init__()
        Config.exception_logger = exception_logger
        Config.enrich_token_usage = enrich_token_usage
        Config.get_common_metrics_attributes = get_common_metrics_attributes

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
            ) = (None, None, None, None)

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
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"groq.resources.completions.{wrap_object}",
                wrapped_method.get("method"),
            )
