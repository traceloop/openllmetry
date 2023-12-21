"""OpenTelemetry Vertex AI instrumentation"""
import logging
import os
import types
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

from opentelemetry.semconv.ai import SpanAttributes, LLMRequestTypeValues
from opentelemetry.instrumentation.vertexai.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("google-cloud-aiplatform >= 1.38.1",)

WRAPPED_METHODS = [
    {
        "package": "vertexai.preview.generative_models",
        "object": "GenerativeModel",
        "method": "generate_content",
        "span_name": "vertexai.generate_content",
    },
    {
        "package": "vertexai.language_models",
        "object": "TextGenerationModel",
        "method": "predict",
        "span_name": "vertexai.predict",
    },
    {
        "package": "vertexai.language_models",
        "object": "TextGenerationModel",
        "method": "predict_async",
        "span_name": "vertexai.predict_async",
    },
    {
        "package": "vertexai.language_models",
        "object": "TextGenerationModel",
        "method": "predict_streaming",
        "span_name": "vertexai.predict_streaming",
    },
    {
        "package": "vertexai.language_models",
        "object": "TextGenerationModel",
        "method": "predict_streaming_async",
        "span_name": "vertexai.predict_streaming_async",
    },
]


def should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def is_streaming_response(response):
    return isinstance(response, types.GeneratorType)


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


input_attribute_map = {
    "prompt": f"{SpanAttributes.LLM_PROMPTS}.0.user",
    "temperature": SpanAttributes.LLM_TEMPERATURE,
    "top_p": SpanAttributes.LLM_TOP_P,
}


def _set_input_attributes(span, args, kwargs):
    if args is not None and len(args) > 0:
        _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, args[0])
    elif kwargs.get("version"):
        _set_span_attribute(
            span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("version").id
        )
    else:
        _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, "unknown")

    input_attribute = kwargs.get("input")
    for key in input_attribute:
        if key in input_attribute_map:
            if key == "prompt" and not should_send_prompts():
                continue
            _set_span_attribute(
                span,
                input_attribute_map.get(key, f"llm.request.{key}"),
                input_attribute.get(key),
            )
    return


def _set_span_completions(span, llm_request_type, completion):
    index = 0
    prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
    _set_span_attribute(span, f"{prefix}.finish_reason", completion.get("stop_reason"))
    _set_span_attribute(span, f"{prefix}.content", completion.get("completion"))


def _set_response_attributes(span, response):
    _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, response.get("model"))
    if should_send_prompts():
        _set_span_completions(span, response)

def _build_from_streaming_response(span, response):
    complete_response = ""
    for item in response:
        item_to_yield = item
        complete_response += str(item)

        yield item_to_yield

    _set_response_attributes(span, complete_response)

    span.set_status(Status(StatusCode.OK))
    span.end()

def _handle_request(span, args, kwargs):
    try:
        if span.is_recording():
            _set_input_attributes(span, args, kwargs)

    except Exception as ex:  # pylint: disable=broad-except
        logger.warning(
            "Failed to set input attributes for vertexai span, error: %s", str(ex)
        )

def _handle_response(span, response):
    try:
        if span.is_recording():
            _set_response_attributes(span, response)

    except Exception as ex:  # pylint: disable=broad-except
        logger.warning(
            "Failed to set response attributes for vertexai span, error: %s",
            str(ex),
        )
    if span.is_recording():
        span.set_status(Status(StatusCode.OK))


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
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_VENDOR: "Replicate",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
    )

    _handle_request(span, args, kwargs)

    response = wrapped(*args, **kwargs)

    if response:
        if is_streaming_response(response):
            return _build_from_streaming_response(span, response)
        else:
            _handle_response(span, response)

    span.end()
    return response


class VertexAiInstrumentor(BaseInstrumentor):
    """An instrumentor for VertextAI's client library."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            wrap_function_wrapper(
                wrap_package,
                f"{wrap_object}.{wrap_method}",
                _wrap(tracer, wrapped_method),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"{wrap_package}.{wrap_object}",
                wrapped_method.get("method", ""),
            )
