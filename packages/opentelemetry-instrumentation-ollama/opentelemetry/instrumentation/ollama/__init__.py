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
from opentelemetry.instrumentation.ollama.utils import dont_throw, should_emit_events
from opentelemetry.instrumentation.ollama.version import __version__
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.metrics import Histogram, Meter, get_meter
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics as GenAIMetrics
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

_instruments = ("ollama >= 0.4.0, < 1",)

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


def _sanitize_copy_messages(wrapped, instance, args, kwargs):
    # original signature: _copy_messages(messages)
    messages = args[0] if args else []
    sanitized = []
    for msg in messages or []:
        if isinstance(msg, dict):
            msg_copy = dict(msg)
            tc_list = msg_copy.get("tool_calls")
            if tc_list:
                for tc in tc_list:
                    func = tc.get("function")
                    arg = func.get("arguments") if func else None
                    if isinstance(arg, str):
                        try:
                            func["arguments"] = json.loads(arg)
                        except Exception:
                            pass
            sanitized.append(msg_copy)
        else:
            sanitized.append(msg)
    return wrapped(sanitized)


def _accumulate_streaming_response(
    span,
    event_logger,
    token_histogram,
    llm_request_type,
    response,
    streaming_time_to_first_token=None,
    streaming_time_to_generate=None,
    start_time=None,
):
    if llm_request_type == LLMRequestTypeValues.CHAT:
        accumulated_response = {"message": {"content": "", "role": ""}}
    elif llm_request_type == LLMRequestTypeValues.COMPLETION:
        accumulated_response = {"response": ""}

    first_token = True
    first_token_time = None
    last_response = None

    for res in response:
        last_response = res  # Track the last response explicitly

        if first_token and streaming_time_to_first_token and start_time is not None:
            first_token_time = time.perf_counter()
            streaming_time_to_first_token.record(
                first_token_time - start_time,
                attributes={SpanAttributes.LLM_SYSTEM: "Ollama"},
            )
            first_token = False
        yield res

        if llm_request_type == LLMRequestTypeValues.CHAT:
            accumulated_response["message"]["content"] += res["message"]["content"]
            accumulated_response["message"]["role"] = res["message"]["role"]
        elif llm_request_type == LLMRequestTypeValues.COMPLETION:
            text = res.get("response", "")
            accumulated_response["response"] += text

    # Record streaming time to generate after the response is complete
    if streaming_time_to_generate and first_token_time is not None:
        model_name = last_response.get("model") if last_response else None
        streaming_time_to_generate.record(
            time.perf_counter() - first_token_time,
            attributes={
                SpanAttributes.LLM_SYSTEM: "Ollama",
                SpanAttributes.LLM_RESPONSE_MODEL: model_name,
            },
        )

    response_data = (
        last_response.model_dump()
        if last_response and hasattr(last_response, 'model_dump')
        else last_response
    )
    _handle_response(
        span=span,
        event_logger=event_logger,
        llm_request_type=llm_request_type,
        token_histogram=token_histogram,
        response=response_data | accumulated_response,
    )
    span.end()


async def _aaccumulate_streaming_response(
    span,
    event_logger,
    token_histogram,
    llm_request_type,
    response,
    streaming_time_to_first_token=None,
    streaming_time_to_generate=None,
    start_time=None,
):
    if llm_request_type == LLMRequestTypeValues.CHAT:
        accumulated_response = {"message": {"content": "", "role": ""}}
    elif llm_request_type == LLMRequestTypeValues.COMPLETION:
        accumulated_response = {"response": ""}

    first_token = True
    first_token_time = None
    last_response = None

    async for res in response:
        last_response = res

        if first_token and streaming_time_to_first_token and start_time is not None:
            first_token_time = time.perf_counter()
            streaming_time_to_first_token.record(
                first_token_time - start_time,
                attributes={SpanAttributes.LLM_SYSTEM: "Ollama"},
            )
            first_token = False
        yield res

        if llm_request_type == LLMRequestTypeValues.CHAT:
            accumulated_response["message"]["content"] += res["message"]["content"]
            accumulated_response["message"]["role"] = res["message"]["role"]
        elif llm_request_type == LLMRequestTypeValues.COMPLETION:
            text = res.get("response", "")
            accumulated_response["response"] += text

    # Record streaming time to generate after the response is complete
    if streaming_time_to_generate and first_token_time is not None:
        model_name = last_response.get("model") if last_response else None
        streaming_time_to_generate.record(
            time.perf_counter() - first_token_time,
            attributes={
                SpanAttributes.LLM_SYSTEM: "Ollama",
                SpanAttributes.LLM_RESPONSE_MODEL: model_name,
            },
        )

    response_data = (
        last_response.model_dump()
        if last_response and hasattr(last_response, 'model_dump')
        else last_response
    )
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
        tracer,
        token_histogram,
        duration_histogram,
        event_logger,
        streaming_time_to_first_token,
        streaming_time_to_generate,
        to_wrap,
    ):
        def wrapper(wrapped, instance, args, kwargs):
            return func(
                tracer,
                token_histogram,
                duration_histogram,
                event_logger,
                streaming_time_to_first_token,
                streaming_time_to_generate,
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
    streaming_time_to_first_token: Histogram,
    streaming_time_to_generate: Histogram,
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
            attrs = {SpanAttributes.LLM_SYSTEM: "Ollama"}
            # Try to get model from response, then fallback to request
            model = None
            if isinstance(response, dict):
                model = response.get("model")
            if not model:
                json_data = kwargs.get("json", {})
                if json_data:
                    model = json_data.get("model")
            if model is not None:
                attrs[SpanAttributes.LLM_RESPONSE_MODEL] = model
            duration_histogram.record(duration, attributes=attrs)

        if kwargs.get("stream"):
            return _accumulate_streaming_response(
                span,
                event_logger,
                token_histogram,
                llm_request_type,
                response,
                streaming_time_to_first_token,
                streaming_time_to_generate,
                start_time,
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
    streaming_time_to_first_token: Histogram,
    streaming_time_to_generate: Histogram,
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
            attrs = {SpanAttributes.LLM_SYSTEM: "Ollama"}
            # Try to get model from response, then fallback to request
            model = None
            if isinstance(response, dict):
                model = response.get("model")
            if not model:
                json_data = kwargs.get("json", {})
                if json_data:
                    model = json_data.get("model")
            if model is not None:
                attrs[SpanAttributes.LLM_RESPONSE_MODEL] = model
            duration_histogram.record(duration, attributes=attrs)

        if kwargs.get("stream"):
            return _aaccumulate_streaming_response(
                span,
                event_logger,
                token_histogram,
                llm_request_type,
                response,
                streaming_time_to_first_token,
                streaming_time_to_generate,
                start_time,
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

    return token_histogram, duration_histogram, streaming_time_to_first_token, streaming_time_to_generate


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
            event_logger_provider = kwargs.get("event_logger_provider")
            event_logger = get_event_logger(
                __name__, __version__, event_logger_provider=event_logger_provider
            )

        # Patch _copy_messages to sanitize tool_calls arguments before Pydantic validation
        wrap_function_wrapper(
            "ollama._client",
            "_copy_messages",
            _sanitize_copy_messages,
        )
        # instrument all llm methods (generate/chat/embeddings) via _request dispatch wrapper
        wrap_function_wrapper(
            "ollama._client",
            "Client._request",
            _dispatch_wrap(
                tracer,
                token_histogram,
                duration_histogram,
                event_logger,
                streaming_time_to_first_token,
                streaming_time_to_generate,
            ),
        )
        wrap_function_wrapper(
            "ollama._client",
            "AsyncClient._request",
            _dispatch_awrap(
                tracer,
                token_histogram,
                duration_histogram,
                event_logger,
                streaming_time_to_first_token,
                streaming_time_to_generate,
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


def _dispatch_wrap(
    tracer,
    token_histogram,
    duration_histogram,
    event_logger,
    streaming_time_to_first_token,
    streaming_time_to_generate
):
    def wrapper(wrapped, instance, args, kwargs):
        to_wrap = None
        if len(args) > 2 and isinstance(args[2], str):
            path = args[2]
            op = path.rstrip("/").split("/")[-1]
            to_wrap = next((m for m in WRAPPED_METHODS if m.get("method") == op), None)
        if to_wrap:
            return _wrap(
                tracer,
                token_histogram,
                duration_histogram,
                event_logger,
                streaming_time_to_first_token,
                streaming_time_to_generate,
                to_wrap,
            )(wrapped, instance, args, kwargs)
        return wrapped(*args, **kwargs)

    return wrapper


def _dispatch_awrap(
    tracer,
    token_histogram,
    duration_histogram,
    event_logger,
    streaming_time_to_first_token,
    streaming_time_to_generate,
):
    async def wrapper(wrapped, instance, args, kwargs):
        to_wrap = None
        if len(args) > 2 and isinstance(args[2], str):
            path = args[2]
            op = path.rstrip("/").split("/")[-1]
            to_wrap = next((m for m in WRAPPED_METHODS if m.get("method") == op), None)
        if to_wrap:
            return await _awrap(
                tracer,
                token_histogram,
                duration_histogram,
                event_logger,
                streaming_time_to_first_token,
                streaming_time_to_generate,
                to_wrap,
            )(wrapped, instance, args, kwargs)
        return await wrapped(*args, **kwargs)

    return wrapper
