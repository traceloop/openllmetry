"""OpenTelemetry Together AI instrumentation"""

import logging
import os
from typing import Collection

from opentelemetry.instrumentation.together.config import Config
from opentelemetry.instrumentation.together.events import message_to_event, choice_to_event
from opentelemetry.instrumentation.together.utils import dont_throw
from wrapt import wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry._events import EventLogger
from opentelemetry.trace import get_tracer, SpanKind
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)

from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
    LLMRequestTypeValues,
)
from opentelemetry.instrumentation.together.version import __version__

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


def should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


@dont_throw
def _set_input_attributes(span, llm_request_type, kwargs):
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
    _set_span_attribute(
        span,
        SpanAttributes.LLM_IS_STREAMING,
        kwargs.get("stream"),
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
        elif llm_request_type == LLMRequestTypeValues.COMPLETION:
            _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_PROMPTS}.0.content", kwargs.get("prompt")
            )


@dont_throw
def _set_response_attributes(span, llm_request_type, response):
    if should_send_prompts():
        if llm_request_type == LLMRequestTypeValues.COMPLETION:
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
                response.choices[0].text,
            )
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant"
            )
        elif llm_request_type == LLMRequestTypeValues.CHAT:
            index = 0
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
            _set_span_attribute(
                span, f"{prefix}.content", response.choices[0].message.content
            )
            _set_span_attribute(
                span, f"{prefix}.role", response.choices[0].message.role
            )

    _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, response.model)

    usage_data = response.usage
    input_tokens = getattr(usage_data, "prompt_tokens", 0)
    output_tokens = getattr(usage_data, "completion_tokens", 0)

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


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, event_logger, to_wrap, config):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, event_logger, to_wrap, config, wrapped, instance, args, kwargs)
        return wrapper
    return _with_tracer


def _llm_request_type_by_method(method_name):
    if method_name == "chat.completions.ChatCompletions.create":
        return LLMRequestTypeValues.CHAT
    elif method_name == "completions.Completions.create":
        return LLMRequestTypeValues.COMPLETION
    else:
        return LLMRequestTypeValues.UNKNOWN


@_with_tracer_wrapper
def _wrap(tracer, event_logger, to_wrap, config, wrapped, instance, args, kwargs):
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
            SpanAttributes.LLM_SYSTEM: "TogetherAI",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    )

    if span.is_recording():
        if config.use_legacy_attributes:
            _set_input_attributes(span, llm_request_type, kwargs)
        else:
            if llm_request_type == LLMRequestTypeValues.CHAT:
                for message in kwargs.get("messages", []):
                    event_logger.emit(
                        message_to_event(message, config.capture_content)
                    )
            elif llm_request_type == LLMRequestTypeValues.COMPLETION:
                event_logger.emit(
                    message_to_event(
                        {"role": "user", "content": kwargs.get("prompt")},
                        config.capture_content
                    )
                )

    try:
        response = wrapped(*args, **kwargs)

        if response and span.is_recording():
            if config.use_legacy_attributes:
                _set_response_attributes(span, llm_request_type, response)
            else:
                for choice in response.choices:
                    event_logger.emit(
                        choice_to_event(choice, config.capture_content)
                    )

            span.set_status(Status(StatusCode.OK))

        return response

    except Exception as ex:
        if span.is_recording():
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(ex)
            if config.exception_logger:
                config.exception_logger(ex)
        raise
    finally:
        span.end()


class TogetherAiInstrumentor(BaseInstrumentor):
    """An instrumentor for Together AI's client library."""

    def __init__(self, exception_logger=None):
        super().__init__()
        self._exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        self._config = Config(
            use_legacy_attributes=kwargs.get("use_legacy_attributes", True),
            capture_content=kwargs.get("capture_content", True),
            exception_logger=self._exception_logger,
        )

        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        event_logger = EventLogger(__name__, tracer_provider)

        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            wrap_function_wrapper(
                "together",
                f"{wrap_object}.{wrap_method}",
                _wrap(tracer, event_logger, wrapped_method, self._config),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            unwrap(
                f"together.{wrap_object}",
                wrap_method.split(".")[-1],
            )
