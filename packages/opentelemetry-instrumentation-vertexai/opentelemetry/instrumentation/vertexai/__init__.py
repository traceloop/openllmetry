"""OpenTelemetry Vertex AI instrumentation"""

import logging
import types
from typing import Collection

from opentelemetry import context as context_api
from opentelemetry._logs import get_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap
from opentelemetry.instrumentation.vertexai.config import Config
from opentelemetry.instrumentation.vertexai.event_emitter import (
    emit_prompt_events,
    emit_response_events,
)
from opentelemetry.instrumentation.vertexai.span_utils import (
    _map_vertex_finish_reason,
    accumulate_vertex_stream_finish_reasons,
    set_input_attributes,
    set_input_attributes_sync,
    set_model_input_attributes,
    set_model_response_attributes,
    set_response_attributes,
)
from opentelemetry.instrumentation.vertexai.utils import dont_throw, should_emit_events
from opentelemetry.instrumentation.vertexai.version import __version__
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
)
from opentelemetry.trace import SpanKind, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

_GCP_VERTEX_AI = GenAIAttributes.GenAiProviderNameValues.GCP_VERTEX_AI.value
_OP_CHAT = GenAIAttributes.GenAiOperationNameValues.CHAT.value
_OP_GENERATE_CONTENT = GenAIAttributes.GenAiOperationNameValues.GENERATE_CONTENT.value
_OP_TEXT_COMPLETION = GenAIAttributes.GenAiOperationNameValues.TEXT_COMPLETION.value

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


def _gen_ai_operation_name(span_name: str) -> str:
    if "predict" in span_name:
        return _OP_TEXT_COMPLETION
    if "send_message" in span_name:
        return _OP_CHAT
    return _OP_GENERATE_CONTENT


def is_streaming_response(response):
    return isinstance(response, types.GeneratorType)


def is_async_streaming_response(response):
    return isinstance(response, types.AsyncGeneratorType)


@dont_throw
def handle_streaming_response(
    span,
    event_logger,
    llm_model,
    complete_response,
    token_usage,
    stream_last_chunk=None,
    stream_finish_reasons=None,
):
    set_model_response_attributes(
        span,
        llm_model,
        token_usage,
        response_meta=stream_last_chunk,
        stream_finish_reasons=stream_finish_reasons,
    )
    finish_reason_otel = None
    if stream_last_chunk and getattr(stream_last_chunk, "candidates", None):
        finish_reason_otel = _map_vertex_finish_reason(
            stream_last_chunk.candidates[0].finish_reason
        )
    if should_emit_events():
        emit_response_events(complete_response, event_logger)
    else:
        # Prefer full candidate on the last chunk when present (see google-generativeai streaming).
        if stream_last_chunk is not None and getattr(
            stream_last_chunk, "candidates", None
        ):
            set_response_attributes(span, llm_model, stream_last_chunk)
        else:
            set_response_attributes(
                span,
                llm_model,
                complete_response,
                finish_reason_otel=finish_reason_otel,
            )
    if span.is_recording():
        span.set_status(Status(StatusCode.OK))


def _build_from_streaming_response(span, event_logger, response, llm_model):
    text_parts = []
    token_usage = None
    last_item = None
    stream_finish_ordered = []
    stream_finish_seen = set()
    for item in response:
        item_to_yield = item
        last_item = item
        t = getattr(item, "text", None)
        if isinstance(t, str):
            text_parts.append(t)
        if item.usage_metadata:
            token_usage = item.usage_metadata
        accumulate_vertex_stream_finish_reasons(
            stream_finish_ordered, stream_finish_seen, item
        )

        yield item_to_yield

    complete_response = "".join(text_parts)

    handle_streaming_response(
        span,
        event_logger,
        llm_model,
        complete_response,
        token_usage,
        stream_last_chunk=last_item,
        stream_finish_reasons=stream_finish_ordered or None,
    )

    span.set_status(Status(StatusCode.OK))
    span.end()


async def _abuild_from_streaming_response(span, event_logger, response, llm_model):
    text_parts = []
    token_usage = None
    last_item = None
    stream_finish_ordered = []
    stream_finish_seen = set()
    async for item in response:
        item_to_yield = item
        last_item = item
        t = getattr(item, "text", None)
        if isinstance(t, str):
            text_parts.append(t)
        if item.usage_metadata:
            token_usage = item.usage_metadata
        accumulate_vertex_stream_finish_reasons(
            stream_finish_ordered, stream_finish_seen, item
        )

        yield item_to_yield

    complete_response = "".join(text_parts)

    handle_streaming_response(
        span,
        event_logger,
        llm_model,
        complete_response,
        token_usage,
        stream_last_chunk=last_item,
        stream_finish_reasons=stream_finish_ordered or None,
    )

    span.set_status(Status(StatusCode.OK))
    span.end()


@dont_throw
async def _handle_request(span, event_logger, args, kwargs, llm_model):
    set_model_input_attributes(span, kwargs, llm_model)
    if should_emit_events():
        emit_prompt_events(args, event_logger)
    else:
        await set_input_attributes(span, args)


def _handle_response(span, event_logger, response, llm_model):
    set_model_response_attributes(
        span, llm_model, getattr(response, "usage_metadata", None), response_meta=response
    )
    if should_emit_events():
        emit_response_events(response, event_logger)
    else:
        set_response_attributes(span, llm_model, response)
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
    # For ChatSession, try to get model from the parent model object
    if hasattr(instance, "_model") and hasattr(instance._model, "_model_name"):
        llm_model = instance._model._model_name.replace("publishers/google/models/", "")
    elif hasattr(instance, "_model") and hasattr(instance._model, "_model_id"):
        llm_model = instance._model._model_id

    name = to_wrap.get("span_name")
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            GenAIAttributes.GEN_AI_PROVIDER_NAME: _GCP_VERTEX_AI,
            GenAIAttributes.GEN_AI_OPERATION_NAME: _gen_ai_operation_name(name),
            GenAIAttributes.GEN_AI_REQUEST_MODEL: llm_model,
        },
    )

    await _handle_request(span, event_logger, args, kwargs, llm_model)

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
    # For ChatSession, try to get model from the parent model object
    if hasattr(instance, "_model") and hasattr(instance._model, "_model_name"):
        llm_model = instance._model._model_name.replace("publishers/google/models/", "")
    elif hasattr(instance, "_model") and hasattr(instance._model, "_model_id"):
        llm_model = instance._model._model_id

    name = to_wrap.get("span_name")
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            GenAIAttributes.GEN_AI_PROVIDER_NAME: _GCP_VERTEX_AI,
            GenAIAttributes.GEN_AI_OPERATION_NAME: _gen_ai_operation_name(name),
            GenAIAttributes.GEN_AI_REQUEST_MODEL: llm_model,
        },
    )

    # Use sync version for non-async wrapper to avoid image processing for now
    set_model_input_attributes(span, kwargs, llm_model)
    if should_emit_events():
        emit_prompt_events(args, event_logger)
    else:
        set_input_attributes_sync(span, args)

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

    def __init__(self, exception_logger=None, use_legacy_attributes=True, upload_base64_image=None):
        super().__init__()
        Config.exception_logger = exception_logger
        Config.use_legacy_attributes = use_legacy_attributes
        if upload_base64_image:
            Config.upload_base64_image = upload_base64_image

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        event_logger = None

        if should_emit_events():
            logger_provider = kwargs.get("logger_provider")
            event_logger = get_logger(
                __name__,
                __version__,
                logger_provider=logger_provider,
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
