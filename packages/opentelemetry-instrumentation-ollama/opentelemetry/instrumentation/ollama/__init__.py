"""OpenTelemetry Ollama instrumentation"""

import json
import logging
import os
import time
from typing import Collection

from opentelemetry import context as context_api
from opentelemetry._events import get_event_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.ollama.config import Config
from opentelemetry.instrumentation.ollama.event_emitter import (
    emit_choice_events,
    emit_message_events,
)
from opentelemetry.instrumentation.ollama.span_utils import (
    set_input_attributes,
    set_model_input_attributes,
    set_model_response_attributes,
    set_response_attributes,
)
from opentelemetry.instrumentation.ollama.utils import (
    dont_throw,
    should_emit_events,
    should_send_prompts,
)
from opentelemetry.instrumentation.ollama.version import __version__
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.metrics import Histogram, Meter, get_meter
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    Meters,
    SpanAttributes,
)
from opentelemetry.trace import SpanKind, Tracer, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("ollama >= 0.2.0, < 1",)

WRAPPED_METHODS = [
    {
        "method": "generate",
        "span_name": "ollama.completion",
    },
    {
        "method": "chat",
        "span_name": "ollama.chat",
    },
    {
        "method": "embeddings",
        "span_name": "ollama.embeddings",
    },
]


def _accumulate_streaming_response(
    span, event_logger, token_histogram, llm_request_type, response
):
    if llm_request_type == LLMRequestTypeValues.CHAT:
        accumulated_response = {"message": {"content": "", "role": ""}}
    elif llm_request_type == LLMRequestTypeValues.COMPLETION:
        accumulated_response = {"response": ""}

    for res in response:
        yield res

        if llm_request_type == LLMRequestTypeValues.CHAT:
            accumulated_response["message"]["content"] += res["message"]["content"]
            accumulated_response["message"]["role"] = res["message"]["role"]
        elif llm_request_type == LLMRequestTypeValues.COMPLETION:
            accumulated_response["response"] += res["response"]
        if res.get("done"):
            accumulated_response["done_reason"] = res.get("done_reason")

    response_data = res.model_dump() if hasattr(res, "model_dump") else res
    _handle_response(
        span,
        event_logger,
        llm_request_type,
        token_histogram,
        response_data | accumulated_response,
    )

    span.end()


async def _aaccumulate_streaming_response(
    span, event_logger, token_histogram, llm_request_type, response
):
    if llm_request_type == LLMRequestTypeValues.CHAT:
        accumulated_response = {"message": {"content": "", "role": ""}}
    elif llm_request_type == LLMRequestTypeValues.COMPLETION:
        accumulated_response = {"response": ""}

    async for res in response:
        yield res

        if llm_request_type == LLMRequestTypeValues.CHAT:
            accumulated_response["message"]["content"] += res["message"]["content"]
            accumulated_response["message"]["role"] = res["message"]["role"]
        elif llm_request_type == LLMRequestTypeValues.COMPLETION:
            accumulated_response["response"] += res["response"]
        if res.get("done"):
            accumulated_response["done_reason"] = res.get("done_reason")

    response_data = res.model_dump() if hasattr(res, "model_dump") else res
    _handle_response(
        span,
        event_logger,
        llm_request_type,
        token_histogram,
        response_data | accumulated_response,
    )

    span.end()


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(
        tracer, token_histogram, duration_histogram, event_logger, to_wrap
    ):
        def wrapper(wrapped, instance, args, kwargs):
            return func(
                tracer,
                token_histogram,
                duration_histogram,
                event_logger,
                to_wrap,
                wrapped,
                instance,
                args,
                kwargs,
            )

        return wrapper

    return _with_tracer


def _llm_request_type_by_method(method_name):
    if method_name == "chat":
        return LLMRequestTypeValues.CHAT
    elif method_name == "generate":
        return LLMRequestTypeValues.COMPLETION
    elif method_name == "embeddings":
        return LLMRequestTypeValues.EMBEDDING
    else:
        return LLMRequestTypeValues.UNKNOWN


@dont_throw
def _handle_input(span, event_logger, llm_request_type, args, kwargs):
    set_model_input_attributes(span, kwargs)
    if should_emit_events() and event_logger:
        emit_message_events(llm_request_type, args, kwargs, event_logger)
    else:
        set_input_attributes(span, llm_request_type, kwargs)


@dont_throw
def _handle_response(span, event_logger, llm_request_type, token_histogram, response):
    if should_emit_events() and event_logger:
        emit_choice_events(llm_request_type, response, event_logger)
    else:
        set_response_attributes(span, token_histogram, llm_request_type, response)

    set_model_response_attributes(span, token_histogram, llm_request_type, response)


@_with_tracer_wrapper
def _wrap(
    tracer: Tracer,
    token_histogram: Histogram,
    duration_histogram: Histogram,
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
            SpanAttributes.LLM_SYSTEM: "Ollama",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    )
    _handle_input(span, event_logger, llm_request_type, args, kwargs)

    start_time = time.perf_counter()
    response = wrapped(*args, **kwargs)
    end_time = time.perf_counter()

    if response:
        if duration_histogram:
            duration = end_time - start_time
            duration_histogram.record(
                duration,
                attributes={
                    SpanAttributes.LLM_SYSTEM: "Ollama",
                    SpanAttributes.LLM_RESPONSE_MODEL: kwargs.get("model"),
                },
            )

        if kwargs.get("stream"):
            return _accumulate_streaming_response(
                span, event_logger, token_histogram, llm_request_type, response
            )
        _handle_response(
            span, event_logger, llm_request_type, token_histogram, response
        )
        if span.is_recording():
            span.set_status(Status(StatusCode.OK))

    span.end()
    return response


@_with_tracer_wrapper
async def _awrap(
    tracer: Tracer,
    token_histogram: Histogram,
    duration_histogram: Histogram,
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
        return await wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    llm_request_type = _llm_request_type_by_method(to_wrap.get("method"))
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Ollama",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    )

    _handle_input(span, event_logger, llm_request_type, args, kwargs)

    start_time = time.perf_counter()
    response = await wrapped(*args, **kwargs)
    end_time = time.perf_counter()
    if response:
        if duration_histogram:
            duration = end_time - start_time
            duration_histogram.record(
                duration,
                attributes={
                    SpanAttributes.LLM_SYSTEM: "Ollama",
                    SpanAttributes.LLM_RESPONSE_MODEL: kwargs.get("model"),
                },
            )

        if kwargs.get("stream"):
            return _aaccumulate_streaming_response(
                span, event_logger, token_histogram, llm_request_type, response
            )

        _handle_response(
            span, event_logger, llm_request_type, token_histogram, response
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
        description="GenAI operation duration",
    )

    return token_histogram, duration_histogram


def is_metrics_collection_enabled() -> bool:
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"


class OllamaInstrumentor(BaseInstrumentor):
    """An instrumentor for Ollama's client library."""

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
            ) = _build_metrics(meter)
        else:
            (
                token_histogram,
                duration_histogram,
            ) = (None, None)

        event_logger = None
        if not Config.use_legacy_attributes:
            event_logger_provider = kwargs.get("event_logger_provider")
            event_logger = get_event_logger(
                __name__, __version__, event_logger_provider=event_logger_provider
            )

        for wrapped_method in WRAPPED_METHODS:
            wrap_method = wrapped_method.get("method")
            wrap_function_wrapper(
                "ollama._client",
                f"Client.{wrap_method}",
                _wrap(
                    tracer,
                    token_histogram,
                    duration_histogram,
                    event_logger,
                    wrapped_method,
                ),
            )
            wrap_function_wrapper(
                "ollama._client",
                f"AsyncClient.{wrap_method}",
                _awrap(
                    tracer,
                    token_histogram,
                    duration_histogram,
                    event_logger,
                    wrapped_method,
                ),
            )
            wrap_function_wrapper(
                "ollama",
                f"{wrap_method}",
                _wrap(
                    tracer,
                    token_histogram,
                    duration_histogram,
                    event_logger,
                    wrapped_method,
                ),
            )

    def _uninstrument(self, **kwargs):
        try:
            import ollama
            from ollama._client import AsyncClient, Client

            for wrapped_method in WRAPPED_METHODS:
                method_name = wrapped_method.get("method")
                unwrap(Client, method_name)
                unwrap(AsyncClient, method_name)
                unwrap(ollama, method_name)
        except ImportError:
            logger.warning("Failed to import ollama modules for uninstrumentation.")
