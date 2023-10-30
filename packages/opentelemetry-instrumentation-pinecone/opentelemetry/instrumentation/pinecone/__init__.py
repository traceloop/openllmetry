"""OpenTelemetry Pinecone instrumentation"""

import logging
import pinecone
from typing import Collection
from wrapt import wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, SpanKind
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.instrumentation.pinecone.version import __version__

from opentelemetry.semconv.ai import SpanAttributes

logger = logging.getLogger(__name__)

_instruments = ("pinecone-client ~= 2.2.2",)

WRAPPED_METHODS = [
    {
        "object": "GRPCIndex",
        "method": "query",
        "span_name": "pinecone.query",
    },
    {
        "object": "GRPCIndex",
        "method": "upsert",
        "span_name": "pinecone.upsert",
    },
    {
        "object": "GRPCIndex",
        "method": "delete",
        "span_name": "pinecone.delete",
    },
    {
        "object": "Index",
        "method": "query",
        "span_name": "pinecone.query",
    },
    {
        "object": "Index",
        "method": "upsert",
        "span_name": "pinecone.upsert",
    },
    {
        "object": "Index",
        "method": "delete",
        "span_name": "pinecone.delete",
    },
]


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _set_input_attributes(span, kwargs):
    pass


def _set_response_attributes(span, response):
    pass


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


@_with_tracer_wrapper
def _wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    with tracer.start_as_current_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.VECTOR_DB_VENDOR: "Pinecone",
        },
    ) as span:
        try:
            if span.is_recording():
                _set_input_attributes(span, kwargs)

        except Exception as ex:  # pylint: disable=broad-except
            logger.warning(
                "Failed to set input attributes for openai span, error: %s", str(ex)
            )

        response = wrapped(*args, **kwargs)

        if response:
            try:
                if span.is_recording():
                    _set_response_attributes(span, response)

            except Exception as ex:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to set response attributes for openai span, error: %s",
                    str(ex),
                )
            if span.is_recording():
                span.set_status(Status(StatusCode.OK))

        return response


class PineconeInstrumentor(BaseInstrumentor):
    """An instrumentor for Pinecone's client library."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            if getattr(pinecone, wrap_object, None):
                wrap_function_wrapper(
                    "pinecone",
                    f"{wrap_object}.{wrap_method}",
                    _wrap(tracer, wrapped_method),
                )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            unwrap(f"pinecone.{wrap_object}", wrapped_method.get("method"))
