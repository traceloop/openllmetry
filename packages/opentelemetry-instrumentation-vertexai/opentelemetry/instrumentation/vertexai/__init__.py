"""OpenTelemetry Vertex AI instrumentation"""

import logging
import types
from typing import Collection

from opentelemetry import context as context_api
from opentelemetry._events import get_event_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap
from opentelemetry.instrumentation.vertexai.config import Config
from opentelemetry.instrumentation.vertexai.event_emitter import (
    emit_prompt_events,
    emit_response_events,
)
from opentelemetry.instrumentation.vertexai.span_utils import (
    set_input_attributes,
    set_model_input_attributes,
    set_model_response_attributes,
    set_response_attributes,
)
from opentelemetry.instrumentation.vertexai.utils import dont_throw, should_emit_events
from opentelemetry.instrumentation.vertexai.version import __version__
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    SpanAttributes,
)
from opentelemetry.trace import SpanKind, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("google-cloud-aiplatform >= 1.38.1",)

WRAPPED_METHODS = [
    {
        "package": "vertexai.generative_models",
        "object": "GenerativeModel",
        "method": "generate_content",
        "span_name": "vertexai.generate_content",
        "is_async": False,
    },
    {
        "package": "vertexai.generative_models",
        "object": "GenerativeModel",
        "method": "generate_content_async",
        "span_name": "vertexai.generate_content_async",
        "is_async": True,
    },
    {
        "package": "vertexai.generative_models",
        "object": "ChatSession",
        "method": "send_message",
        "span_name": "vertexai.send_message",
        "is_async": False,
    },
    {
        "package": "vertexai.preview.generative_models",
        "object": "GenerativeModel",
        "method": "generate_content",
        "span_name": "vertexai.generate_content",
        "is_async": False,
    },
    {
        "package": "vertexai.preview.generative_models",
        "object": "GenerativeModel",
        "method": "generate_content_async",
        "span_name": "vertexai.generate_content_async",
        "is_async": True,
    },
    {
        "package": "vertexai.preview.generative_models",
        "object": "ChatSession",
        "method": "send_message",
        "span_name": "vertexai.send_message",
        "is_async": False,
    },
    {
        "package": "vertexai.language_models",
        "object": "TextGenerationModel",
        "method": "predict",
        "span_name": "vertexai.predict",
        "is_async": False,
    },
    {
        "package": "vertexai.language_models",
        "object": "TextGenerationModel",
        "method": "predict_async",
        "span_name": "vertexai.predict_async",
        "is_async": True,
    },
    {
        "package": "vertexai.language_models",
        "object": "TextGenerationModel",
        "method": "predict_streaming",
        "span_name": "vertexai.predict_streaming",
        "is_async": False,
    },
    {
        "package": "vertexai.language_models",
        "object": "TextGenerationModel",
        "method": "predict_streaming_async",
        "span_name": "vertexai.predict_streaming_async",
        "is_async": True,
    },
    {
        "package": "vertexai.language_models",
        "object": "ChatSession",
        "method": "send_message",
        "span_name": "vertexai.send_message",
        "is_async": False,
    },
    {
        "package": "vertexai.language_models",
        "object": "ChatSession",
        "method": "send_message_streaming",
        "span_name": "vertexai.send_message_streaming",
        "is_async": False,
    },
]


def is_streaming_response(response):
    return isinstance(response, types.GeneratorType)


def is_async_streaming_response(response):
    return isinstance(response, types.AsyncGeneratorType)


@dont_throw
def handle_streaming_response(span, event_logger, llm_model, response, token_usage):
    set_model_response_attributes(span, llm_model, token_usage)
    if should_emit_events():
        emit_response_events(response, event_logger)
    else:
        set_response_attributes(span, llm_model, response)
    if span.is_recording():
        span.set_status(Status(StatusCode.OK))


def _build_from_streaming_response(span, event_logger, response, llm_model):
    complete_response = ""
    token_usage = None
    for item in response:
        item_to_yield = item
        complete_response += str(item.text)
        if item.usage_metadata:
            token_usage = item.usage_metadata

        yield item_to_yield

    handle_streaming_response(
        span, event_logger, llm_model, complete_response, token_usage
    )

    span.set_status(Status(StatusCode.OK))
    span.end()


async def _abuild_from_streaming_response(span, event_logger, response, llm_model):
    complete_response = ""
    token_usage = None
    async for item in response:
        item_to_yield = item
        complete_response += str(item.text)
        if item.usage_metadata:
            token_usage = item.usage_metadata

        yield item_to_yield

    handle_streaming_response(span, event_logger, llm_model, response, token_usage)

    span.set_status(Status(StatusCode.OK))
    span.end()


@dont_throw
def _handle_request(span, event_logger, args, kwargs, llm_model):
    set_model_input_attributes(span, kwargs, llm_model)
    if should_emit_events():
        emit_prompt_events(args, event_logger)
    else:
        set_input_attributes(span, args)


def _handle_response(span, event_logger, response, llm_model):
    set_model_response_attributes(span, llm_model, response.usage_metadata)
    if should_emit_events():
        emit_response_events(response, event_logger)
    else:
        set_response_attributes(
            span, llm_model, response.candidates[0].text if response.candidates else ""
        )
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
async def _awrap(tracer, event_logger, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return await wrapped(*args, **kwargs)

    llm_model = "unknown"
    if hasattr(instance, "_model_id"):
        llm_model = instance._model_id
    if hasattr(instance, "_model_name"):
        llm_model = instance._model_name.replace("publishers/google/models/", "")

    name = to_wrap.get("span_name")
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Google",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
    )

    _handle_request(span, event_logger, args, kwargs, llm_model)

    response = await wrapped(*args, **kwargs)

    if response:
        if is_streaming_response(response):
            return _build_from_streaming_response(
                span, event_logger, response, llm_model
            )
        elif is_async_streaming_response(response):
            return _abuild_from_streaming_response(
                span, event_logger, response, llm_model
            )
        else:
            _handle_response(span, event_logger, response, llm_model)

    span.end()
    return response


@_with_tracer_wrapper
def _wrap(tracer, event_logger, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    llm_model = "unknown"
    if hasattr(instance, "_model_id"):
        llm_model = instance._model_id
    if hasattr(instance, "_model_name"):
        llm_model = instance._model_name.replace("publishers/google/models/", "")

    name = to_wrap.get("span_name")
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Google",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
    )

    _handle_request(span, event_logger, args, kwargs, llm_model)

    response = wrapped(*args, **kwargs)

    if response:
        if is_streaming_response(response):
            return _build_from_streaming_response(
                span, event_logger, response, llm_model
            )
        elif is_async_streaming_response(response):
            return _abuild_from_streaming_response(
                span, event_logger, response, llm_model
            )
        else:
            _handle_response(span, event_logger, response, llm_model)

    span.end()
    return response


class VertexAIInstrumentor(BaseInstrumentor):
    """An instrumentor for VertextAI's client library."""

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
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            wrap_function_wrapper(
                wrap_package,
                f"{wrap_object}.{wrap_method}",
                (
                    _awrap(tracer, event_logger, wrapped_method)
                    if wrapped_method.get("is_async")
                    else _wrap(tracer, event_logger, wrapped_method)
                ),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"{wrap_package}.{wrap_object}",
                wrapped_method.get("method", ""),
            )
