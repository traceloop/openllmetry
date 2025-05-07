"""OpenTelemetry Aleph Alpha instrumentation"""

import logging
import os
from typing import Collection, List, Optional, Union

from opentelemetry import context as context_api
from opentelemetry._events import EventLogger, get_event_logger
from opentelemetry.instrumentation.alephalpha.config import Config
from opentelemetry.instrumentation.alephalpha.event_handler import (
    ChoiceEvent,
    MessageEvent,
    emit_event,
)
from opentelemetry.instrumentation.alephalpha.utils import dont_throw
from opentelemetry.instrumentation.alephalpha.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    SpanAttributes,
)
from opentelemetry.trace import SpanKind, Tracer, get_tracer
from opentelemetry.trace.span import Span
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("aleph_alpha_client >= 7.1.0, <8",)

WRAPPED_METHODS = [
    {
        "method": "complete",
        "span_name": "alephalpha.completion",
    },
]

TRACELOOP_TRACE_CONTENT = "TRACELOOP_TRACE_CONTENT"


def should_send_prompts():
    return (
        os.getenv(TRACELOOP_TRACE_CONTENT) or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def should_emit_events():
    return not Config.use_legacy_attributes


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _set_prompt_attributes(event: MessageEvent, span: Span):
    if not span.is_recording():
        return

    if should_send_prompts():
        _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_PROMPTS}.0.content",
            event.content[0].get("data"),
        )


def _set_completion_attributes(event: ChoiceEvent, span: Span):
    if not span.is_recording():
        return

    if should_send_prompts():
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
            event.message["content"],
        )
        _set_span_attribute(
            span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant"
        )


@dont_throw
def _handle_message_event(
    event: MessageEvent, span: Span, event_logger: Optional[EventLogger], kwargs
):
    if span.is_recording():
        _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))

    if should_emit_events():
        return emit_event(event, event_logger)
    else:
        return _set_prompt_attributes(event, span)


def _handle_response_event(event: ChoiceEvent, span, event_logger, response):
    if span.is_recording():
        input_tokens = getattr(response, "num_tokens_prompt_total", 0)
        output_tokens = getattr(response, "num_tokens_generated", 0)

        _set_span_attribute(
            span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, input_tokens + output_tokens
        )
        _set_span_attribute(
            span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, output_tokens
        )
        _set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, input_tokens)

    if should_emit_events():
        emit_event(event, event_logger)
    else:
        _set_completion_attributes(event, span)


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, event_logger, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, event_logger, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def _llm_request_type_by_method(method_name):
    if method_name == "complete":
        return LLMRequestTypeValues.COMPLETION
    else:
        return LLMRequestTypeValues.UNKNOWN


def _parse_message_event(args, kwargs) -> MessageEvent:
    request = kwargs.get("request") if kwargs.get("request") else args[0]

    return MessageEvent(
        content=request.prompt.to_json(),
        role="user",
    )


def _parse_response_event(response) -> List[ChoiceEvent]:
    return ChoiceEvent(
        index=0,
        message={
            "content": response.completions[0].completion,
            "role": "assistant",
        },
        finish_reason=response.completions[0].finish_reason,
    )


@_with_tracer_wrapper
def _wrap(
    tracer: Tracer,
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
            SpanAttributes.LLM_SYSTEM: "AlephAlpha",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    )
    input_event = _parse_message_event(args, kwargs)
    _handle_message_event(input_event, span, event_logger, kwargs)

    response = wrapped(*args, **kwargs)

    if response:
        response_event = _parse_response_event(response)
        _handle_response_event(response_event, span, event_logger, response)
        if span.is_recording():
            span.set_status(Status(StatusCode.OK))

    span.end()
    return response


class AlephAlphaInstrumentor(BaseInstrumentor):
    """An instrumentor for Aleph Alpha's client library."""

    def __init__(self, exception_logger=None, use_legacy_attributes=True):
        super().__init__()
        Config.exception_logger = exception_logger
        Config.use_legacy_attributes = use_legacy_attributes

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        event_logger = None

        if should_emit_events():
            event_logger_provider = kwargs.get("event_logger_provider")
            event_logger = get_event_logger(
                __name__,
                __version__,
                event_logger_provider=event_logger_provider,
            )

        for wrapped_method in WRAPPED_METHODS:
            wrap_method = wrapped_method.get("method")
            wrap_function_wrapper(
                "aleph_alpha_client",
                f"Client.{wrap_method}",
                _wrap(tracer, event_logger, wrapped_method),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            unwrap(
                "aleph_alpha_client.Client",
                wrapped_method.get("method"),
            )
