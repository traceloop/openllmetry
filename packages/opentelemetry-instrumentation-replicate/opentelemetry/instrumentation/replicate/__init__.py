"""OpenTelemetry Replicate instrumentation"""

import logging
import os
from typing import Collection

from opentelemetry.instrumentation.replicate.config import Config
from opentelemetry.instrumentation.replicate.utils import dont_throw
from opentelemetry.instrumentation.replicate.events import (
    prompt_to_event,
    completion_to_event,
)
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
from opentelemetry.instrumentation.replicate.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("replicate >= 0.22.0",)

WRAPPED_METHODS = [
    {
        "method": "run",
        "span_name": "replicate.run",
        "streaming": False,
    },
    {
        "method": "predictions.create",
        "span_name": "replicate.predictions.create",
        "streaming": False,
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
def _set_input_attributes(span, model, input_data):
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, model)

    if should_send_prompts():
        if isinstance(input_data, dict):
            for key, value in input_data.items():
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{key}",
                    str(value) if value is not None else None,
                )
        else:
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.0.content",
                str(input_data) if input_data is not None else None,
            )


@dont_throw
def _set_response_attributes(span, model, response):
    if should_send_prompts():
        if isinstance(response, (list, dict)):
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
                str(response),
            )
        else:
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
                str(response) if response is not None else None,
            )

    _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, model)


def _accumulate_streaming_response(span, event_logger, model, response, config):
    """Accumulate streaming response and set attributes."""
    accumulated_response = []

    for res in response:
        yield res
        accumulated_response.append(res)

        if not config.use_legacy_attributes:
            event_logger.emit(
                completion_to_event(res, model, config.capture_content)
            )

    if config.use_legacy_attributes:
        _set_response_attributes(span, model, accumulated_response)
    span.end()


async def _aaccumulate_streaming_response(span, event_logger, model, response, config):
    """Accumulate streaming response and set attributes."""
    accumulated_response = []

    async for res in response:
        yield res
        accumulated_response.append(res)

        if not config.use_legacy_attributes:
            event_logger.emit(
                completion_to_event(res, model, config.capture_content)
            )

    if config.use_legacy_attributes:
        _set_response_attributes(span, model, accumulated_response)
    span.end()


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, event_logger, to_wrap, config):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, event_logger, to_wrap, config, wrapped, instance, args, kwargs)
        return wrapper
    return _with_tracer


@_with_tracer_wrapper
def _wrap(tracer, event_logger, to_wrap, config, wrapped, instance, args, kwargs):
    """Wrap a synchronous Replicate method with tracing."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    model = args[0] if args else kwargs.get("model")
    input_data = args[1] if len(args) > 1 else kwargs.get("input")

    with tracer.start_as_current_span(
        to_wrap["span_name"],
        kind=SpanKind.CLIENT,
        end_on_exit=not kwargs.get("stream", False),
    ) as span:
        if span.is_recording():
            if config.use_legacy_attributes:
                _set_input_attributes(span, model, input_data)
            else:
                event_logger.emit(
                    prompt_to_event(input_data, model, config.capture_content)
                )

        try:
            response = wrapped(*args, **kwargs)

            if span.is_recording():
                if kwargs.get("stream", False):
                    return _accumulate_streaming_response(
                        span, event_logger, model, response, config
                    )

                if config.use_legacy_attributes:
                    _set_response_attributes(span, model, response)
                else:
                    event_logger.emit(
                        completion_to_event(response, model, config.capture_content)
                    )

            return response

        except Exception as ex:
            if span.is_recording():
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(ex)
            raise


@_with_tracer_wrapper
async def _awrap(tracer, event_logger, to_wrap, config, wrapped, instance, args, kwargs):
    """Wrap an asynchronous Replicate method with tracing."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    model = args[0] if args else kwargs.get("model")
    input_data = args[1] if len(args) > 1 else kwargs.get("input")

    with tracer.start_as_current_span(
        to_wrap["span_name"],
        kind=SpanKind.CLIENT,
        end_on_exit=not kwargs.get("stream", False),
    ) as span:
        if span.is_recording():
            if config.use_legacy_attributes:
                _set_input_attributes(span, model, input_data)
            else:
                event_logger.emit(
                    prompt_to_event(input_data, model, config.capture_content)
                )

        try:
            response = await wrapped(*args, **kwargs)

            if span.is_recording():
                if kwargs.get("stream", False):
                    return _aaccumulate_streaming_response(
                        span, event_logger, model, response, config
                    )

                if config.use_legacy_attributes:
                    _set_response_attributes(span, model, response)
                else:
                    event_logger.emit(
                        completion_to_event(response, model, config.capture_content)
                    )

            return response

        except Exception as ex:
            if span.is_recording():
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(ex)
            raise


class ReplicateInstrumentor(BaseInstrumentor):
    """An instrumentor for Replicate's client library."""

    def __init__(self, exception_logger=None):
        self._exception_logger = exception_logger
        super().__init__()

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
            wrap_function_wrapper(
                "replicate",
                f"Client.{wrapped_method['method']}",
                _wrap(tracer, event_logger, wrapped_method, self._config),
            )
            wrap_function_wrapper(
                "replicate",
                f"AsyncClient.{wrapped_method['method']}",
                _awrap(tracer, event_logger, wrapped_method, self._config),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            unwrap(
                "replicate",
                f"Client.{wrapped_method['method']}",
            )
            unwrap(
                "replicate",
                f"AsyncClient.{wrapped_method['method']}",
            )
