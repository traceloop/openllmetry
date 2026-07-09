"""OpenTelemetry Cohere instrumentation"""

import logging
from typing import Collection, Union

from opentelemetry import context as context_api
from opentelemetry._logs import Logger, get_logger
from opentelemetry.instrumentation.cohere.config import Config
from opentelemetry.instrumentation.cohere.event_emitter import (
    emit_input_event,
    emit_response_events,
)
from opentelemetry.instrumentation.cohere.span_utils import (
    set_input_content_attributes,
    set_response_content_attributes,
    set_span_request_attributes,
    set_span_response_attributes,
)
from opentelemetry.instrumentation.cohere.streaming import (
    process_chat_v1_streaming_response,
    aprocess_chat_v1_streaming_response,
    process_chat_v2_streaming_response,
    aprocess_chat_v2_streaming_response,
)
from opentelemetry.instrumentation.cohere.utils import dont_throw, should_emit_events
from opentelemetry.instrumentation.cohere.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    SpanAttributes,
)
from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer, get_tracer, use_span
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("cohere >=4.2.7, <6",)

WRAPPED_METHODS = [
    {
        "module": "cohere.client",
        "object": "Client",
        "method": "generate",
        "span_name": "cohere.completion",
    },
    {
        "module": "cohere.client",
        "object": "Client",
        "method": "chat",
        "span_name": "cohere.chat",
    },
    {
        "module": "cohere.client",
        "object": "Client",
        "method": "chat_stream",
        "span_name": "cohere.chat",
        "stream_process_func": process_chat_v1_streaming_response,
    },
    {
        "module": "cohere.client",
        "object": "Client",
        "method": "rerank",
        "span_name": "cohere.rerank",
    },
    {
        "module": "cohere.client",
        "object": "Client",
        "method": "embed",
        "span_name": "cohere.embed",
    },
    {
        "module": "cohere.client_v2",
        "object": "ClientV2",
        "method": "chat",
        "span_name": "cohere.chat",
    },
    {
        "module": "cohere.client_v2",
        "object": "ClientV2",
        "method": "chat_stream",
        "span_name": "cohere.chat",
        "stream_process_func": process_chat_v2_streaming_response,
    },
    {
        "module": "cohere.client_v2",
        "object": "ClientV2",
        "method": "rerank",
        "span_name": "cohere.rerank",
    },
    {
        "module": "cohere.client_v2",
        "object": "ClientV2",
        "method": "embed",
        "span_name": "cohere.embed",
    },
    # Async methods that return AsyncIterator must be wrapped with sync wrapper
    {
        "module": "cohere.client",
        "object": "AsyncClient",
        "method": "chat_stream",
        "span_name": "cohere.chat",
        "stream_process_func": aprocess_chat_v1_streaming_response,
    },
    {
        "module": "cohere.client_v2",
        "object": "AsyncClientV2",
        "method": "chat_stream",
        "span_name": "cohere.chat",
        "stream_process_func": aprocess_chat_v2_streaming_response,
    },
]

WRAPPED_AMETHODS = [
    {
        "module": "cohere.client",
        "object": "AsyncClient",
        "method": "generate",
        "span_name": "cohere.completion",
    },
    {
        "module": "cohere.client",
        "object": "AsyncClient",
        "method": "chat",
        "span_name": "cohere.chat",
    },
    {
        "module": "cohere.client",
        "object": "AsyncClient",
        "method": "rerank",
        "span_name": "cohere.rerank",
    },
    {
        "module": "cohere.client",
        "object": "AsyncClient",
        "method": "embed",
        "span_name": "cohere.embed",
    },
    {
        "module": "cohere.client_v2",
        "object": "AsyncClientV2",
        "method": "chat",
        "span_name": "cohere.chat",
    },
    {
        "module": "cohere.client_v2",
        "object": "AsyncClientV2",
        "method": "rerank",
        "span_name": "cohere.rerank",
    },
    {
        "module": "cohere.client_v2",
        "object": "AsyncClientV2",
        "method": "embed",
        "span_name": "cohere.embed",
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
    if method_name in ["chat", "chat_stream"]:
        return LLMRequestTypeValues.CHAT
    elif method_name in ["generate", "generate_stream"]:
        return LLMRequestTypeValues.COMPLETION
    elif method_name == "rerank":
        return LLMRequestTypeValues.RERANK
    elif method_name == "embed":
        return LLMRequestTypeValues.EMBEDDING
    else:
        return LLMRequestTypeValues.UNKNOWN


@dont_throw
def _handle_input_content(span, event_logger, llm_request_type, kwargs):
    set_input_content_attributes(span, llm_request_type, kwargs)
    if should_emit_events():
        emit_input_event(event_logger, llm_request_type, kwargs)


@dont_throw
def _handle_response_content(span, event_logger, llm_request_type, response):
    set_response_content_attributes(span, llm_request_type, response)
    if should_emit_events():
        emit_response_events(event_logger, llm_request_type, response)


@_with_tracer_wrapper
def _wrap(
    tracer: Tracer,
    event_logger: Union[Logger, None],
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
            SpanAttributes.LLM_SYSTEM: "Cohere",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    )

    with use_span(span, end_on_exit=False):
        set_span_request_attributes(span, kwargs)
        _handle_input_content(span, event_logger, llm_request_type, kwargs)

        try:
            response = wrapped(*args, **kwargs)
        except Exception as e:
            if span.is_recording():
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.end()
            raise

        if to_wrap.get("stream_process_func"):
            return to_wrap.get("stream_process_func")(span, event_logger, llm_request_type, response)

        set_span_response_attributes(span, response)
        _handle_response_content(span, event_logger, llm_request_type, response)
        span.end()
        return response


@_with_tracer_wrapper
async def _awrap(
    tracer: Tracer,
    event_logger: Union[Logger, None],
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
    with tracer.start_as_current_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            GenAIAttributes.GEN_AI_SYSTEM: "Cohere",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    ) as span:
        set_span_request_attributes(span, kwargs)
        _handle_input_content(span, event_logger, llm_request_type, kwargs)

        try:
            response = await wrapped(*args, **kwargs)
        except Exception as e:
            if span.is_recording():
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.end()
            raise

        set_span_response_attributes(span, response)
        _handle_response_content(span, event_logger, llm_request_type, response)

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
            logger_provider = kwargs.get("logger_provider")
            event_logger = get_logger(
                __name__, __version__, logger_provider=logger_provider
            )
        for wrapped_method in WRAPPED_METHODS:
            wrap_module = wrapped_method.get("module")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            try:
                wrap_function_wrapper(
                    wrap_module,
                    f"{wrap_object}.{wrap_method}",
                    _wrap(tracer, event_logger, wrapped_method),
                )
            except (ImportError, ModuleNotFoundError, AttributeError):
                logger.debug(f"Failed to instrument {wrap_module}.{wrap_object}.{wrap_method}")

        for wrapped_method in WRAPPED_AMETHODS:
            wrap_module = wrapped_method.get("module")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            try:
                wrap_function_wrapper(
                    wrap_module,
                    f"{wrap_object}.{wrap_method}",
                    _awrap(tracer, event_logger, wrapped_method),
                )
            except (ImportError, ModuleNotFoundError, AttributeError):
                logger.debug(f"Failed to instrument {wrap_module}.{wrap_object}.{wrap_method}")

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_module = wrapped_method.get("module")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            try:
                unwrap(
                    f"{wrap_module}.{wrap_object}",
                    wrap_method,
                )
            except (ImportError, ModuleNotFoundError, AttributeError):
                logger.debug(f"Failed to uninstrument {wrap_module}.{wrap_object}.{wrap_method}")
        for wrapped_method in WRAPPED_AMETHODS:
            wrap_module = wrapped_method.get("module")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            try:
                unwrap(
                    f"{wrap_module}.{wrap_object}",
                    wrap_method,
                )
            except (ImportError, ModuleNotFoundError, AttributeError):
                logger.debug(f"Failed to uninstrument {wrap_module}.{wrap_object}.{wrap_method}")
