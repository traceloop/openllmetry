"""OpenTelemetry Groq instrumentation"""

import json
import logging
import os
import time
from typing import Callable, Collection
import sys
from wrapt import unwrap_function_wrapper

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

from opentelemetry.instrumentation.groq import WRAPPED_METHODS, WRAPPED_AMETHODS  # Import here
from opentelemetry.trace import Span, SpanKind, Tracer, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper, unwrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("groq >= 0.9.0",)

CONTENT_FILTER_KEY = "content_filter_results"

import logging
from typing import Optional

# First, let's fix the module paths in the WRAPPED_METHODS
WRAPPED_METHODS = [
    {
    
     "package": "groq.resources.chat.completions",  # Correct
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

def _emit_prompt_event(span: Span, role: str, content: str, index: int):
    print(f"[_emit_prompt_event] Role: {role}, Content: {content}, Index: {index}")
    """
    Emits a prompt event with standardized attributes.
    
    Args:
        span: The OpenTelemetry span to add the event to
        role: The role of the message sender (e.g., "user", "assistant")
        content: The content of the message
        index: The position of this message in the sequence
    """
    if not content:
        return
    
    attributes = {
        "messaging.role": role,
        "messaging.content": content,
        "messaging.index": index
    }
    span.add_event("prompt", attributes=attributes)

def _emit_completion_event(span: Span, content: str, index: int, usage: Optional[dict] = None):
    print(f"[_emit_completion_event] Content: {content}, Index: {index}, Usage: {usage}")
    """
    Emits a completion event with standardized attributes.
    
    Args:
        span: The OpenTelemetry span to add the event to
        content: The completion content
        index: The index of this completion
        usage: Optional token usage statistics
    """
    if not content:
        return
        
    attributes = {
        "messaging.content": content,
        "messaging.index": index
    }
    
    if usage:
        attributes.update({
            "llm.usage.total_tokens": usage.get("total_tokens"),
            "llm.usage.prompt_tokens": usage.get("prompt_tokens"),
            "llm.usage.completion_tokens": usage.get("completion_tokens")
        })
    
    span.add_event("completion", attributes=attributes)



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
# Fix for _set_response_attributes in __init__.py
def _set_response_attributes(span: Span, response: dict):
    """
    Sets response attributes and emits completion events.
    
    Args:
        span: The OpenTelemetry span to update
        response: The response from the Groq API
    """
    if not span.is_recording():
        return
        
    # Set basic response attributes
    set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, response.get("model"))
    
    # Handle usage information
    usage = response.get("usage", {})
    if usage:
        set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.get("total_tokens"))
        set_span_attribute(span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, usage.get("completion_tokens"))
        set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.get("prompt_tokens"))
    
    if not should_send_prompts():
        return
        
    

    choices = response.get("choices", [])

    for choice in choices:
        message = choice.get("message", {})
        if not message:
            continue

        index = choice.get("index", 0)
        content = message.get("content")

        if Config.use_legacy_attributes:
            # Set attributes in the legacy format
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
            set_span_attribute(span, f"{prefix}.role", message.get("role"))
            set_span_attribute(span, f"{prefix}.content", content)
            set_span_attribute(span, f"{prefix}.finish_reason", choice.get("finish_reason"))
        else:
            # Emit an event with the completion information
            _emit_completion_event(
                span,
                content,
                index,
                usage  # Include usage information in the event
            )


def _set_completions(span, choices):
    if choices is None:
        return

    for choice in choices:
        index = choice.get("index")
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"

        message = choice.get("message")
        if not message:
            continue

        # Always emit the completion event (new behavior)
        content = message.get("content")
        _emit_completion_event(span, content, index)

        # Set legacy attributes only if use_legacy_attributes is True
        if Config.use_legacy_attributes:
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

            set_span_attribute(span, f"{prefix}.role", message.get("role"))
            set_span_attribute(span, f"{prefix}.content", content)

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
# Fix for the module path issue in GroqInstrumentor._uninstrument
def _set_response_attributes(span: Span, response: dict):
    """
    Sets response attributes and emits completion events. This function handles both legacy attributes
    and the new event-based approach, depending on configuration.

    Args:
        span: The OpenTelemetry span to update
        response: The response from the Groq API containing completion and usage data
    """
    # First, set the basic response attributes that are always needed
    set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, response.get("model"))

    # Extract and process usage information
    usage = response.get("usage", {})
    if usage:
        set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.get("total_tokens"))
        set_span_attribute(span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, usage.get("completion_tokens"))
        set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.get("prompt_tokens"))

    # Only proceed with completions if we should send prompts
    if not should_send_prompts():
        return
        
    choices = response.get("choices", [])

    for choice in choices:
        message = choice.get("message", {})
        if not message:
            continue
            
        index = choice.get("index", 0)
        content = message.get("content")
        
        if Config.use_legacy_attributes:
            # Set attributes in the legacy format
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
            set_span_attribute(span, f"{prefix}.role", message.get("role"))
            set_span_attribute(span, f"{prefix}.content", content)
            set_span_attribute(span, f"{prefix}.finish_reason", choice.get("finish_reason"))
        else:
            # Emit an event with the completion information
            _emit_completion_event(
                span,
                content,
                index,
                usage  # Include usage information in the event
            )


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

    # def _uninstrument(self, **kwargs):
    #     """
    #     Uninstruments the Groq client library using the correct module paths.
    #     """
    #     for wrapped_method in WRAPPED_METHODS + WRAPPED_AMETHODS:
    #         package = wrapped_method.get("package")
    #         object_name = wrapped_method.get("object")
    #         method_name = wrapped_method.get("method")
            
    #         try:
    #             unwrap(
    #                 f"{package}.{object_name}",
    #                 method_name
    #             )
    #             logger.debug(f"Successfully uninstrumented {package}.{object_name}.{method_name}")
    #         except Exception as e:
    #             logger.warning(
    #                 f"Failed to uninstrument {package}.{object_name}.{method_name}: {str(e)}. "
    #                 "This is expected if the module was never imported."
    #             )