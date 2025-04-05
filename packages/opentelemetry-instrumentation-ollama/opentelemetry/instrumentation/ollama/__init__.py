"""OpenTelemetry Ollama instrumentation"""

import json
import logging
import os
import time
from typing import Collection, Dict, List, Union

from opentelemetry import context as context_api
from opentelemetry._events import Event, EventLogger, get_event_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.ollama.config import Config
from opentelemetry.instrumentation.ollama.utils import dont_throw, is_content_enabled
from opentelemetry.instrumentation.ollama.version import __version__
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.metrics import Histogram, Meter, get_meter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
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


def should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _set_prompts(span, messages):
    if not span.is_recording() or messages is None:
        return
    for i, msg in enumerate(messages):
        prefix = f"{SpanAttributes.LLM_PROMPTS}.{i}"

        _set_span_attribute(span, f"{prefix}.role", msg.get("role"))
        if msg.get("content"):
            content = msg.get("content")
            if isinstance(content, list):
                content = json.dumps(content)
            _set_span_attribute(span, f"{prefix}.content", content)
        if msg.get("tool_call_id"):
            _set_span_attribute(span, f"{prefix}.tool_call_id", msg.get("tool_call_id"))
        tool_calls = msg.get("tool_calls")
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

                if function.get("arguments"):
                    function["arguments"] = json.loads(function.get("arguments"))


def set_tools_attributes(span, tools):
    if not tools:
        return

    for i, tool in enumerate(tools):
        function = tool.get("function")
        if not function:
            continue

        prefix = f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}"
        _set_span_attribute(span, f"{prefix}.name", function.get("name"))
        _set_span_attribute(span, f"{prefix}.description", function.get("description"))
        _set_span_attribute(
            span, f"{prefix}.parameters", json.dumps(function.get("parameters"))
        )


@dont_throw
def _set_input_attributes(span, llm_request_type, kwargs):
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
    _set_span_attribute(
        span, SpanAttributes.LLM_IS_STREAMING, kwargs.get("stream") or False
    )

    if should_send_prompts():
        if llm_request_type == LLMRequestTypeValues.CHAT:
            _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
            for index, message in enumerate(kwargs.get("messages")):
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{index}.content",
                    message.get("content"),
                )
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{index}.role",
                    message.get("role"),
                )
            _set_prompts(span, kwargs.get("messages"))
            if kwargs.get("tools"):
                set_tools_attributes(span, kwargs.get("tools"))
        else:
            _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_PROMPTS}.0.content", kwargs.get("prompt")
            )


@dont_throw
def _set_response_attributes(span, token_histogram, llm_request_type, response):
    if should_send_prompts():
        if llm_request_type == LLMRequestTypeValues.COMPLETION:
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
                response.get("response"),
            )
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant"
            )
        elif llm_request_type == LLMRequestTypeValues.CHAT:
            index = 0
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
            _set_span_attribute(
                span, f"{prefix}.content", response.get("message").get("content")
            )
            _set_span_attribute(
                span, f"{prefix}.role", response.get("message").get("role")
            )

    if llm_request_type == LLMRequestTypeValues.EMBEDDING:
        return

    _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, response.get("model"))

    input_tokens = response.get("prompt_eval_count") or 0
    output_tokens = response.get("eval_count") or 0

    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
        input_tokens + output_tokens,
    )
    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
        output_tokens,
    )
    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
        input_tokens,
    )
    _set_span_attribute(span, SpanAttributes.LLM_SYSTEM, "Ollama")

    if (
        token_histogram is not None
        and isinstance(input_tokens, int)
        and input_tokens >= 0
    ):
        token_histogram.record(
            input_tokens,
            attributes={
                SpanAttributes.LLM_SYSTEM: "Ollama",
                SpanAttributes.LLM_TOKEN_TYPE: "input",
                SpanAttributes.LLM_RESPONSE_MODEL: response.get("model"),
            },
        )

    if (
        token_histogram is not None
        and isinstance(output_tokens, int)
        and output_tokens >= 0
    ):
        token_histogram.record(
            output_tokens,
            attributes={
                SpanAttributes.LLM_SYSTEM: "Ollama",
                SpanAttributes.LLM_TOKEN_TYPE: "output",
                SpanAttributes.LLM_RESPONSE_MODEL: response.get("model"),
            },
        )


def _accumulate_streaming_response(
    span, token_histogram, llm_request_type, event_logger, response
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
    _set_response_attributes(
        span, token_histogram, llm_request_type, response_data | accumulated_response
    )
    if not Config.use_legacy_attributes and event_logger is not None:
        _emit_response_event(event_logger, llm_request_type, accumulated_response)

    span.end()


async def _aaccumulate_streaming_response(
    span, token_histogram, llm_request_type, event_logger, response
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
    _set_response_attributes(
        span, token_histogram, llm_request_type, response_data | accumulated_response
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
def _emit_input_events(event_logger: EventLogger, llm_request_type, args, kwargs):
    attributes = {GenAIAttributes.GEN_AI_SYSTEM: "ollama"}

    if llm_request_type == LLMRequestTypeValues.CHAT:
        messages: List[Dict] = (
            kwargs.get("messages") if kwargs.get("messages") is not None else args[1]
        )
        for message in messages:
            role = message.get("role")
            name = (
                f"gen_ai.{role}.message" if role is not None else "gen_ai.user.message"
            )
            if is_content_enabled():
                content = message.get("content")
                images = message.get("images")
                tool_calls = message.get("tool_calls")
                body = {
                    "content": {"content": content},
                }
                if images is not None:
                    body["content"]["images"] = images
                if tool_calls is not None:
                    body["tool_calls"] = tool_calls
                if role is not None and role != "user":
                    body["role"] = role
            else:
                body = {}
            event_logger.emit(Event(name=name, attributes=attributes, body=body))

    elif llm_request_type == LLMRequestTypeValues.COMPLETION:
        prompt = kwargs.get("prompt") if kwargs.get("prompt") is not None else args[1]
        body = {"content": prompt} if is_content_enabled() else {}
        event_logger.emit(
            Event(name="gen_ai.user.message", attributes=attributes, body=body)
        )

    elif llm_request_type == LLMRequestTypeValues.EMBEDDING:
        prompt = kwargs.get("prompt") if kwargs.get("prompt") is not None else args[1]
        body = {"content": prompt} if is_content_enabled() else {}
        event_logger.emit(
            Event(name="gen_ai.user.message", attributes=attributes, body=body)
        )

    else:
        raise ValueError(
            "It wasn't possible to emit the input events due to an unknow llm_request_type."
        )


@dont_throw
def _emit_response_event(event_logger: EventLogger, llm_request_type, response: dict):
    attributes = {GenAIAttributes.GEN_AI_SYSTEM: "ollama"}

    if llm_request_type == LLMRequestTypeValues.CHAT:
        finish_reason = response.get("done_reason")
        body = {
            "index": 0,
            "finish_reason": finish_reason if finish_reason is not None else "unknown",
            "message": {},
        }
        if is_content_enabled():
            role = response.get("message").get("role")
            body["message"]["content"] = response.get("message").get("content")
            if role is not None and role != "assistant":
                body["message"]["role"] = role
        event_logger.emit(Event("gen_ai.choice", attributes=attributes, body=body))

    elif llm_request_type == LLMRequestTypeValues.COMPLETION:
        finish_reason = response.get("done_reason")
        body = {
            "index": 0,
            "finish_reason": finish_reason if finish_reason is not None else "unknown",
            "message": {},
        }
        if is_content_enabled():
            body["message"]["content"] = response.get("response")
        event_logger.emit(Event("gen_ai.choice", attributes=attributes, body=body))

    elif llm_request_type == LLMRequestTypeValues.EMBEDDING:
        body = {
            "index": 0,
            "finish_reason": "unknown",
            "message": {},
        }
        if is_content_enabled():
            body["message"]["content"] = response.get("embedding")
        event_logger.emit(Event("gen_ai.choice", attributes=attributes, body=body))

    else:
        raise ValueError(
            "It wasn't possible to emit the choice events due to an unknow llm_request_type."
        )


@_with_tracer_wrapper
def _wrap(
    tracer: Tracer,
    token_histogram: Histogram,
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
    llm_request_type = _llm_request_type_by_method(to_wrap.get("method"))
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Ollama",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    )
    if span.is_recording():
        _set_input_attributes(span, llm_request_type, kwargs)
    if not Config.use_legacy_attributes and event_logger is not None:
        _emit_input_events(event_logger, llm_request_type, args, kwargs)

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

        if span.is_recording():
            if kwargs.get("stream"):
                return _accumulate_streaming_response(
                    span, token_histogram, llm_request_type, event_logger, response
                )
            _set_response_attributes(span, token_histogram, llm_request_type, response)
            if not Config.use_legacy_attributes and event_logger is not None:
                _emit_response_event(event_logger, llm_request_type, response)
            span.set_status(Status(StatusCode.OK))

    span.end()
    return response


@_with_tracer_wrapper
async def _awrap(
    tracer: Tracer,
    token_histogram: Histogram,
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
    llm_request_type = _llm_request_type_by_method(to_wrap.get("method"))
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Ollama",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    )

    if span.is_recording():
        _set_input_attributes(span, llm_request_type, kwargs)
    if not Config.use_legacy_attributes and event_logger is not None:
        _emit_input_events(event_logger, llm_request_type, args, kwargs)

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

        if span.is_recording():
            if kwargs.get("stream"):
                return _aaccumulate_streaming_response(
                    span, token_histogram, llm_request_type, event_logger, response
                )

            _set_response_attributes(span, token_histogram, llm_request_type, response)
            if not Config.use_legacy_attributes and event_logger is not None:
                _emit_response_event(event_logger, llm_request_type, response)
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

        if Config.use_legacy_attributes:
            event_logger = None
        else:
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
