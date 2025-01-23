"""OpenTelemetry Groq instrumentation"""

import json
import logging
import os
import time
from typing import Callable, Collection, Optional

from groq._streaming import AsyncStream, Stream
from opentelemetry import context as context_api
from opentelemetry.instrumentation.groq.config import Config
from opentelemetry.instrumentation.groq.events import (
    create_prompt_event,
    create_completion_event,
    create_function_call_event,
    create_tool_call_event,
)
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
def _set_input_attributes(span, kwargs, event_logger=None, use_legacy_attributes=True):
    if use_legacy_attributes:
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
            if use_legacy_attributes:
                set_span_attribute(
                    span, f"{SpanAttributes.LLM_PROMPTS}.0.user", kwargs.get("prompt")
                )
            if event_logger:
                event_logger.emit_event(create_prompt_event(
                    content=kwargs.get("prompt"),
                    role="user",
                    model=kwargs.get("model"),
                ))

        elif kwargs.get("messages") is not None:
            for i, message in enumerate(kwargs.get("messages")):
                content = _dump_content(message.get("content"))
                if use_legacy_attributes:
                    set_span_attribute(
                        span,
                        f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                        content,
                    )
                    set_span_attribute(
                        span, f"{SpanAttributes.LLM_PROMPTS}.{i}.role", message.get("role")
                    )
                if event_logger:
                    event_logger.emit_event(create_prompt_event(
                        content=content,
                        role=message.get("role"),
                        model=kwargs.get("model"),
                    ))


def _set_completions(span, choices, event_logger=None, use_legacy_attributes=True, model=None):
    if choices is None:
        return

    for choice in choices:
        index = choice.get("index")
        if use_legacy_attributes:
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
                continue

            message = choice.get("message")
            if not message:
                continue

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

        if event_logger:
            message = choice.get("message", {})
            if choice.get("finish_reason") == "content_filter":
                event_logger.emit_event(create_completion_event(
                    completion="FILTERED",
                    model=model,
                    role="assistant",
                    finish_reason=choice.get("finish_reason"),
                    content_filter_results=choice.get("content_filter_results"),
                ))
            else:
                event_logger.emit_event(create_completion_event(
                    completion=message.get("content", ""),
                    model=model,
                    role=message.get("role"),
                    finish_reason=choice.get("finish_reason"),
                ))

                function_call = message.get("function_call")
                if function_call:
                    event_logger.emit_event(create_function_call_event(
                        function_name=function_call.get("name"),
                        function_args=json.loads(function_call.get("arguments", "{}")),
                        model=model,
                    ))

                tool_calls = message.get("tool_calls")
                if tool_calls:
                    for tool_call in tool_calls:
                        function = tool_call.get("function", {})
                        event_logger.emit_event(create_tool_call_event(
                            tool_name=function.get("name"),
                            tool_input=json.loads(function.get("arguments", "{}")),
                            model=model,
                        ))


@dont_throw
def _set_response_attributes(span, response, event_logger=None, use_legacy_attributes=True):
    response = model_as_dict(response)

    if use_legacy_attributes:
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
        _set_completions(span, choices, event_logger, use_legacy_attributes, response.get("model"))


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer: Tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def _with_chat_telemetry_wrapper(func):
    """Helper for providing tracer for wrapper functions. Includes metric collectors."""

    def _with_chat_telemetry(
        tracer: Tracer,
        token_histogram: Histogram,
        choice_counter: Counter,
        duration_histogram: Histogram,
        exception_counter: Counter,
        to_wrap,
    ):
        def wrapper(wrapped, instance, args, kwargs):
            return func(
                tracer,
                token_histogram,
                choice_counter,
                duration_histogram,
                exception_counter,
                to_wrap,
                wrapped,
                instance,
                args,
                kwargs,
            )

        return wrapper

    return _with_chat_telemetry


@_with_chat_telemetry_wrapper
def _wrap(
    tracer: Tracer,
    token_histogram: Histogram,
    choice_counter: Counter,
    duration_histogram: Histogram,
    exception_counter: Counter,
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

    event_logger = kwargs.pop("event_logger", None)
    use_legacy_attributes = kwargs.pop("use_legacy_attributes", True)

    _set_input_attributes(span, kwargs, event_logger, use_legacy_attributes)

    start_time = time.time()
    try:
        result = wrapped(*args, **kwargs)

        if result:
            _set_response_attributes(span, result, event_logger, use_legacy_attributes)

            # Record metrics if enabled
            end_time = time.time()
            metric_attributes = shared_metrics_attributes(result)

            duration_histogram.record(
                end_time - start_time,
                attributes=metric_attributes,
            )

            if result.get("usage"):
                usage = result["usage"]
                if "total_tokens" in usage:
                    token_histogram.record(
                        usage["total_tokens"],
                        attributes=metric_attributes,
                    )

            if result.get("choices"):
                choice_counter.add(
                    len(result["choices"]),
                    attributes=metric_attributes,
                )

        span.set_status(Status(StatusCode.OK))
        return result
    except Exception as ex:
        span.set_status(Status(StatusCode.ERROR, str(ex)))
        end_time = time.time()
        duration_histogram.record(
            end_time - start_time,
            attributes=error_metrics_attributes(ex),
        )
        exception_counter.add(1, attributes=error_metrics_attributes(ex))
        raise
    finally:
        span.end()


def _awrap(tracer, event_logger, wrapped, token_histogram, choice_counter):
    """Helper for wrapping async functions with telemetry."""
    async def wrapper(wrapped, instance, args, kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
        ):
            return await wrapped(*args, **kwargs)

        span = tracer.start_span(
            "groq.chat",
            kind=SpanKind.CLIENT,
            attributes={
                SpanAttributes.LLM_SYSTEM: "Groq",
                SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
            },
        )

        use_legacy_attributes = kwargs.pop("use_legacy_attributes", True)

        _set_input_attributes(span, kwargs, event_logger, use_legacy_attributes)

        start_time = time.time()
        try:
            result = await wrapped(*args, **kwargs)

            if result:
                _set_response_attributes(span, result, event_logger, use_legacy_attributes)

                # Record metrics
                end_time = time.time()
                metric_attributes = shared_metrics_attributes(result)

                if result.get("usage"):
                    usage = result["usage"]
                    if "total_tokens" in usage:
                        token_histogram.record(
                            usage["total_tokens"],
                            attributes=metric_attributes,
                        )

                if result.get("choices"):
                    choice_counter.add(
                        len(result["choices"]),
                        attributes=metric_attributes,
                    )

            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise
        finally:
            span.end()

    return wrapper


def is_metrics_enabled() -> bool:
    """Check if metrics collection is enabled."""
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"


def _create_metrics(meter: Meter):
    """Create metrics collectors.
    
    Args:
        meter: The OpenTelemetry meter to use for creating metrics.
        
    Returns:
        Tuple of (token_histogram, choice_counter, duration_histogram, exception_counter)
    """
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

    exception_counter = meter.create_counter(
        name=Meters.LLM_EXCEPTIONS,
        unit="exception",
        description="Number of exceptions raised during LLM operations",
    )

    return token_histogram, choice_counter, duration_histogram, exception_counter


class GroqInstrumentor(BaseInstrumentor):
    """An instrumentor for Groq's client library."""

    def __init__(
        self,
        enrich_token_usage: bool = False,
        exception_logger=None,
        get_common_metrics_attributes: Callable[[], dict] = lambda: {},
    ):
        """Initialize the instrumentor.

        Args:
            enrich_token_usage: Whether to enrich spans with token usage information.
            exception_logger: Optional callback for logging exceptions.
            get_common_metrics_attributes: Optional callback for getting common attributes
                to be added to all metrics.
        """
        super().__init__()
        Config.exception_logger = exception_logger
        Config.enrich_token_usage = enrich_token_usage
        Config.get_common_metrics_attributes = get_common_metrics_attributes
        self.exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        """Instruments Groq client library."""
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        # meter and counters are inited here
        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        if is_metrics_enabled():
            token_histogram, choice_counter, duration_histogram, exception_counter = _create_metrics(meter)
        else:
            token_histogram, choice_counter, duration_histogram, exception_counter = None, None, None, None

        # Wrap synchronous methods
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
                        exception_counter,
                        wrapped_method,
                    ),
                )
            except ModuleNotFoundError:
                pass  # that's ok, we don't want to fail if some methods do not exist

        # Wrap asynchronous methods
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
                        exception_counter,
                        wrapped_method,
                    ),
                )
            except ModuleNotFoundError:
                # Skip if method doesn't exist (optional features)
                pass

    def _uninstrument(self, **kwargs):
        """Removes instrumentation from Groq client library."""
        for wrapped_method in WRAPPED_METHODS + WRAPPED_AMETHODS:
            unwrap(
                f"{wrapped_method['package']}.{wrapped_method['object']}",
                wrapped_method["method"],
            )

        for wrapped_method in WRAPPED_AMETHODS:
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"groq.resources.completions.{wrap_object}",
                wrapped_method.get("method"),
            )
