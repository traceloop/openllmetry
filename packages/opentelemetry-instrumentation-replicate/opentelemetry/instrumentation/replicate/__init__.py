"""OpenTelemetry Replicate instrumentation"""

import logging
import types
from typing import Collection

from opentelemetry import context as context_api
from opentelemetry._events import get_event_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.replicate.config import Config
from opentelemetry.instrumentation.replicate.event_emitter import (
    emit_choice_events,
    emit_event,
)
from opentelemetry.instrumentation.replicate.event_models import MessageEvent
from opentelemetry.instrumentation.replicate.span_utils import (
    set_input_attributes,
    set_model_input_attributes,
    set_response_attributes,
)
from opentelemetry.instrumentation.replicate.utils import dont_throw, should_emit_events
from opentelemetry.instrumentation.replicate.version import __version__
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from opentelemetry.trace import SpanKind, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("replicate >= 0.22.0",)

WRAPPED_METHODS = [
    {
        "module": "replicate",
        "method": "run",
        "span_name": "replicate.run",
    },
    {
        "module": "replicate",
        "method": "stream",
        "span_name": "replicate.stream",
    },
    {
        "module": "replicate",
        "method": "predictions.create",
        "span_name": "replicate.predictions.create",
    },
]


def is_streaming_response(response):
    return isinstance(response, types.GeneratorType)


def _build_from_streaming_response(span, event_logger, response):
    complete_response = ""
    for item in response:
        item_to_yield = item
        complete_response += str(item)

        yield item_to_yield

    _handle_response(span, event_logger, complete_response)

    span.end()


@dont_throw
def _handle_request(span, event_logger, args, kwargs):
    set_model_input_attributes(span, args, kwargs)

    model_input = kwargs.get("input") or (args[1] if len(args) > 1 else None)

    if should_emit_events() and event_logger:
        emit_event(MessageEvent(content=model_input.get("prompt")), event_logger)
    else:
        set_input_attributes(span, args, kwargs)


@dont_throw
def _handle_response(span, event_logger, response):
    if should_emit_events() and event_logger:
        emit_choice_events(response, event_logger)
    else:
        set_response_attributes(span, response)

    if span.is_recording():
        span.set_status(Status(StatusCode.OK))


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, event_logger, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, event_logger, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


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
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Replicate",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
    )

    _handle_request(span, event_logger, args, kwargs)

    response = wrapped(*args, **kwargs)

    if response:
        if is_streaming_response(response):
            return _build_from_streaming_response(span, event_logger, response)
        else:
            _handle_response(span, event_logger, response)

    span.end()
    return response


class ReplicateInstrumentor(BaseInstrumentor):
    """An instrumentor for Replicate's client library."""

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

        for wrapper_method in WRAPPED_METHODS:
            wrap_function_wrapper(
                wrapper_method.get("module"),
                wrapper_method.get("method"),
                _wrap(tracer, event_logger, wrapper_method),
            )

    def _uninstrument(self, **kwargs):
        import replicate

        for wrapper_method in WRAPPED_METHODS:
            unwrap(replicate, wrapper_method.get("method", ""))
