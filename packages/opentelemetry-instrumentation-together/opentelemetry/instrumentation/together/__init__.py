"""OpenTelemetry Together AI instrumentation"""

import logging
from typing import Collection

from opentelemetry import context as context_api
from opentelemetry._events import get_event_logger
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


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, event_logger, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, event_logger, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def _llm_request_type_by_method(method_name):
    if method_name == "chat.completions.ChatCompletions.create":
        return LLMRequestTypeValues.CHAT
    elif method_name == "completions.Completions.create":
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
            SpanAttributes.LLM_SYSTEM: "TogetherAI",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    )
    _handle_input(span, event_logger, llm_request_type, kwargs)

    response = wrapped(*args, **kwargs)

    if response:
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
            event_logger_provider = kwargs.get("event_logger_provider")
            event_logger = get_event_logger(
                __name__, __version__, event_logger_provider=event_logger_provider
            )

        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            wrap_function_wrapper(
                "together",
                f"{wrap_object}.{wrap_method}",
                _wrap(tracer, event_logger, wrapped_method),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"together.{wrap_object}",
                wrapped_method.get("method"),
            )
