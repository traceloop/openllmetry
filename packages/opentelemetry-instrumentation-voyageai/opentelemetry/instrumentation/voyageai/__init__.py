"""OpenTelemetry Voyage AI instrumentation"""

import logging
from typing import Collection

from opentelemetry import context as context_api
from opentelemetry.instrumentation.voyageai.config import Config
from opentelemetry.instrumentation.voyageai.span_utils import (
    OPERATION_EMBEDDINGS,
    OPERATION_RERANK,
    set_input_content_attributes,
    set_response_content_attributes,
    set_span_request_attributes,
    set_span_response_attributes,
)
from opentelemetry.instrumentation.voyageai.utils import dont_throw
from opentelemetry.instrumentation.voyageai.version import __version__
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
)
from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer, get_tracer
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("voyageai >= 0.3.0",)

WRAPPED_METHODS = [
    {
        "module": "voyageai",
        "object": "Client",
        "method": "embed",
        "span_name": "voyageai.embed",
        "operation_name": OPERATION_EMBEDDINGS,
    },
    {
        "module": "voyageai",
        "object": "Client",
        "method": "rerank",
        "span_name": "voyageai.rerank",
        "operation_name": OPERATION_RERANK,
    },
]

WRAPPED_AMETHODS = [
    {
        "module": "voyageai",
        "object": "AsyncClient",
        "method": "embed",
        "span_name": "voyageai.embed",
        "operation_name": OPERATION_EMBEDDINGS,
    },
    {
        "module": "voyageai",
        "object": "AsyncClient",
        "method": "rerank",
        "span_name": "voyageai.rerank",
        "operation_name": OPERATION_RERANK,
    },
]


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


@dont_throw
def _handle_input_content(span, operation_name, kwargs):
    set_input_content_attributes(span, operation_name, kwargs)


@dont_throw
def _handle_response_content(span, operation_name, response):
    set_response_content_attributes(span, operation_name, response)


@_with_tracer_wrapper
def _wrap(
    tracer: Tracer,
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
    operation_name = to_wrap.get("operation_name")
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            GenAIAttributes.GEN_AI_SYSTEM: "voyageai",
            GenAIAttributes.GEN_AI_OPERATION_NAME: operation_name,
        },
    )

    try:
        set_span_request_attributes(span, kwargs)
        _handle_input_content(span, operation_name, kwargs)

        response = wrapped(*args, **kwargs)

        set_span_response_attributes(span, operation_name, response)
        _handle_response_content(span, operation_name, response)
        span.end()
        return response
    except Exception as e:
        if span.is_recording():
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
        span.end()
        raise


@_with_tracer_wrapper
async def _awrap(
    tracer: Tracer,
    to_wrap,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Instruments and calls every async function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return await wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    operation_name = to_wrap.get("operation_name")
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            GenAIAttributes.GEN_AI_SYSTEM: "voyageai",
            GenAIAttributes.GEN_AI_OPERATION_NAME: operation_name,
        },
    )

    try:
        set_span_request_attributes(span, kwargs)
        _handle_input_content(span, operation_name, kwargs)

        response = await wrapped(*args, **kwargs)

        set_span_response_attributes(span, operation_name, response)
        _handle_response_content(span, operation_name, response)
        span.end()
        return response
    except Exception as e:
        if span.is_recording():
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
        span.end()
        raise


class VoyageAIInstrumentor(BaseInstrumentor):
    """An instrumentor for Voyage AI's client library."""

    def __init__(self, exception_logger=None, use_legacy_attributes=True):
        super().__init__()
        Config.exception_logger = exception_logger
        Config.use_legacy_attributes = use_legacy_attributes

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        for wrapped_method in WRAPPED_METHODS:
            wrap_module = wrapped_method.get("module")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            try:
                wrap_function_wrapper(
                    wrap_module,
                    f"{wrap_object}.{wrap_method}",
                    _wrap(tracer, wrapped_method),
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
                    _awrap(tracer, wrapped_method),
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
