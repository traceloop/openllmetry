"""OpenTelemetry Cohere instrumentation"""

import logging
from typing import Collection, Union

from opentelemetry import context as context_api
from opentelemetry._events import EventLogger, get_event_logger
from opentelemetry.instrumentation.cohere.config import Config
from opentelemetry.instrumentation.cohere.event_emitter import (
    emit_input_event,
    emit_response_events,
)
from opentelemetry.instrumentation.cohere.span_utils import (
    set_input_attributes,
    set_response_attributes,
    set_span_request_attributes,
)
from opentelemetry.instrumentation.cohere.utils import dont_throw, should_emit_events
from opentelemetry.instrumentation.cohere.version import __version__
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
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("cohere >=4.2.7, <6",)

WRAPPED_METHODS = [
    {
        "object": "Client",
        "method": "generate",
        "span_name": "cohere.completion",
    },
    {
        "object": "Client",
        "method": "chat",
        "span_name": "cohere.chat",
    },
    {
        "object": "Client",
        "method": "rerank",
        "span_name": "cohere.rerank",
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
    if method_name == "chat":
        return LLMRequestTypeValues.CHAT
    elif method_name == "generate":
        return LLMRequestTypeValues.COMPLETION
    elif method_name == "rerank":
        return LLMRequestTypeValues.RERANK
    else:
        return LLMRequestTypeValues.UNKNOWN


@dont_throw
def _handle_input(span, event_logger, llm_request_type, kwargs):
    if should_emit_events():
        emit_input_event(event_logger, llm_request_type, kwargs)
    else:
        set_input_attributes(span, llm_request_type, kwargs)


@dont_throw
def _handle_response(span, event_logger, llm_request_type, response):
    if should_emit_events():
        emit_response_events(event_logger, llm_request_type, response)
    else:
        set_response_attributes(span, llm_request_type, response)


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
    with tracer.start_as_current_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Cohere",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    ) as span:
        set_span_request_attributes(span, kwargs)
        _handle_input(span, event_logger, llm_request_type, kwargs)

        response = wrapped(*args, **kwargs)

        if response:
            _handle_response(span, event_logger, llm_request_type, response)

        return response


class CohereInstrumentor(BaseInstrumentor):
    """An instrumentor for Cohere's client library."""

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
        if not Config.use_legacy_attributes:
            event_logger_provider = kwargs.get("event_logger_provider")
            event_logger = get_event_logger(
                __name__, __version__, event_logger_provider=event_logger_provider
            )
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            wrap_function_wrapper(
                "cohere.client",
                f"{wrap_object}.{wrap_method}",
                _wrap(tracer, event_logger, wrapped_method),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"cohere.client.{wrap_object}",
                wrapped_method.get("method"),
            )
