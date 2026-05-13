"""OpenTelemetry Together AI instrumentation"""

import logging
from typing import Collection

from opentelemetry import context as context_api
from opentelemetry._logs import get_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.together.config import Config
from opentelemetry.instrumentation.together.event_emitter import (
    emit_completion_event,
    emit_prompt_events,
)
from opentelemetry.instrumentation.together.span_utils import (
    set_completion_attributes,
    set_model_completion_attributes,
    set_model_prompt_attributes,
    set_prompt_attributes,
)
from opentelemetry.instrumentation.together.utils import dont_throw, should_emit_events
from opentelemetry.instrumentation.together.version import __version__
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    SpanAttributes,
)
from opentelemetry.trace import SpanKind, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("together >= 1.2.0, <2",)

WRAPPED_METHODS = [
    {
        "object": "resources",
        "method": "chat.completions.ChatCompletions.create",
        "span_name": "together.chat",
    },
    {
        "object": "resources",
        "method": "completions.Completions.create",
        "span_name": "together.completion",
    },
]

WRAPPED_AMETHODS = [
    {
        "object": "resources",
        "method": "chat.completions.AsyncChatCompletions.create",
        "span_name": "together.chat",
    },
    {
        "object": "resources",
        "method": "completions.AsyncCompletions.create",
        "span_name": "together.completion",
    },
]


def _is_streaming_response(response):
    """Check if the response is a streaming iterator."""
    return hasattr(response, "__iter__") and not hasattr(response, "choices")


def _is_async_streaming_response(response):
    """Check if the response is an async streaming iterator."""
    return hasattr(response, "__aiter__")


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, event_logger, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, event_logger, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def _llm_request_type_by_method(method_name):
    if "chat.completions" in method_name:
        return LLMRequestTypeValues.CHAT
    elif "completions" in method_name:
        return LLMRequestTypeValues.COMPLETION
    else:
        return LLMRequestTypeValues.UNKNOWN


@dont_throw
def _handle_input(span, event_logger, llm_request_type, kwargs):
    set_model_prompt_attributes(span, kwargs)

    if should_emit_events() and event_logger:
        emit_prompt_events(event_logger, llm_request_type, kwargs)
    else:
        set_prompt_attributes(span, llm_request_type, kwargs)


@dont_throw
def _handle_response(span, event_logger, llm_request_type, response):
    if should_emit_events() and event_logger:
        emit_completion_event(event_logger, llm_request_type, response)
    else:
        set_completion_attributes(span, llm_request_type, response)

    set_model_completion_attributes(span, response)


def _process_streaming_chunk(chunk):
    """Extract content and finish_reason from a streaming chunk."""
    content = ""
    finish_reason = None

    if hasattr(chunk, "choices") and chunk.choices:
        for choice in chunk.choices:
            if hasattr(choice, "delta") and choice.delta:
                delta_content = getattr(choice.delta, "content", None)
                if delta_content:
                    content += delta_content
            if choice.finish_reason:
                finish_reason = str(choice.finish_reason.value) if hasattr(
                    choice.finish_reason, "value"
                ) else str(choice.finish_reason)

    return content, finish_reason


def _build_accumulated_response(chunk, accumulated_content, finish_reason):
    """Build a response-like object from accumulated streaming data for attribute setting."""

    class _AccumulatedChoice:
        def __init__(self, content, finish_reason):
            self.message = type("obj", (object,), {"content": content, "role": "assistant"})()
            self.finish_reason = finish_reason

    class _AccumulatedResponse:
        def __init__(self, chunk, content, finish_reason):
            self.model = getattr(chunk, "model", None)
            self.id = getattr(chunk, "id", None)
            self.choices = [_AccumulatedChoice(content, finish_reason)]
            self.usage = getattr(chunk, "usage", None)

    return _AccumulatedResponse(chunk, accumulated_content, finish_reason)


def _create_stream_processor(response, span, event_logger, llm_request_type):
    """Create a generator that processes a stream while collecting telemetry."""
    accumulated_content = ""
    finish_reason = None
    last_chunk = None

    try:
        for chunk in response:
            last_chunk = chunk
            content, chunk_finish_reason = _process_streaming_chunk(chunk)
            if content:
                accumulated_content += content
            if chunk_finish_reason:
                finish_reason = chunk_finish_reason
            yield chunk
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        span.end()
        raise

    # Build accumulated response and set attributes
    if last_chunk:
        accumulated_response = _build_accumulated_response(
            last_chunk, accumulated_content, finish_reason
        )
        _handle_response(span, event_logger, llm_request_type, accumulated_response)

    if span.is_recording():
        span.set_status(Status(StatusCode.OK))
    span.end()


async def _create_async_stream_processor(response, span, event_logger, llm_request_type):
    """Create an async generator that processes a stream while collecting telemetry."""
    accumulated_content = ""
    finish_reason = None
    last_chunk = None

    try:
        async for chunk in response:
            last_chunk = chunk
            content, chunk_finish_reason = _process_streaming_chunk(chunk)
            if content:
                accumulated_content += content
            if chunk_finish_reason:
                finish_reason = chunk_finish_reason
            yield chunk
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        span.end()
        raise

    # Build accumulated response and set attributes
    if last_chunk:
        accumulated_response = _build_accumulated_response(
            last_chunk, accumulated_content, finish_reason
        )
        _handle_response(span, event_logger, llm_request_type, accumulated_response)

    if span.is_recording():
        span.set_status(Status(StatusCode.OK))
    span.end()


@_with_tracer_wrapper
def _wrap(
    tracer,
    event_logger,
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
    llm_request_type = _llm_request_type_by_method(to_wrap.get("method"))
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            GenAIAttributes.GEN_AI_SYSTEM: "TogetherAI",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    )
    _handle_input(span, event_logger, llm_request_type, kwargs)

    try:
        response = wrapped(*args, **kwargs)
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        span.end()
        raise

    if response:
        if kwargs.get("stream", False) and _is_streaming_response(response):
            return _create_stream_processor(
                response, span, event_logger, llm_request_type
            )

        _handle_response(span, event_logger, llm_request_type, response)
        if span.is_recording():
            span.set_status(Status(StatusCode.OK))

    span.end()
    return response


@_with_tracer_wrapper
async def _awrap(
    tracer,
    event_logger,
    to_wrap,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Instruments and calls every async function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return await wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    llm_request_type = _llm_request_type_by_method(to_wrap.get("method"))
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            GenAIAttributes.GEN_AI_SYSTEM: "TogetherAI",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    )
    _handle_input(span, event_logger, llm_request_type, kwargs)

    try:
        response = await wrapped(*args, **kwargs)
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        span.end()
        raise

    if response:
        if kwargs.get("stream", False) and _is_async_streaming_response(response):
            return _create_async_stream_processor(
                response, span, event_logger, llm_request_type
            )

        _handle_response(span, event_logger, llm_request_type, response)
        if span.is_recording():
            span.set_status(Status(StatusCode.OK))

    span.end()
    return response


class TogetherAiInstrumentor(BaseInstrumentor):
    """An instrumentor for Together AI's client library."""

    def __init__(self, exception_logger=None, use_legacy_attributes: bool = True):
        super().__init__()
        Config.exception_logger = exception_logger
        Config.use_legacy_attributes = use_legacy_attributes

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        event_logger = None
        if not Config.use_legacy_attributes:
            logger_provider = kwargs.get("logger_provider")
            event_logger = get_logger(
                __name__, __version__, logger_provider=logger_provider
            )

        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            wrap_function_wrapper(
                "together",
                f"{wrap_object}.{wrap_method}",
                _wrap(tracer, event_logger, wrapped_method),
            )

        for wrapped_method in WRAPPED_AMETHODS:
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            try:
                wrap_function_wrapper(
                    "together",
                    f"{wrap_object}.{wrap_method}",
                    _awrap(tracer, event_logger, wrapped_method),
                )
            except (ModuleNotFoundError, AttributeError):
                pass  # async methods may not be available in all versions

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"together.{wrap_object}",
                wrapped_method.get("method"),
            )
        for wrapped_method in WRAPPED_AMETHODS:
            wrap_object = wrapped_method.get("object")
            try:
                unwrap(
                    f"together.{wrap_object}",
                    wrapped_method.get("method"),
                )
            except (ModuleNotFoundError, AttributeError):
                pass
