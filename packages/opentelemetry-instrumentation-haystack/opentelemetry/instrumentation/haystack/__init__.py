import logging
from typing import Collection
from wrapt import wrap_function_wrapper
import openai

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, SpanKind
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)

from opentelemetry.semconv.llm import SpanAttributes, LLMRequestTypeValues

logger = logging.getLogger(__name__)

_instruments = ("farm-haystack >= 1.20.1",)
__version__ = "0.1.0"

WRAPPED_METHODS = [
    {
        "package": "haystack.nodes.prompt.invocation_layer.chatgpt",
        "object": "ChatGPTInvocationLayer",
        "method": "_execute_openai_request",
        "span_name": "openai.chat",
    },
    {
        "package": "haystack.nodes.prompt.invocation_layer.open_ai",
        "object": "OpenAIInvocationLayer",
        "method": "_execute_openai_request",
        "span_name": "openai.completion",
    },
]


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _set_input_attributes(span, llm_request_type, kwargs):
    base_payload = kwargs.get("base_payload")
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MODEL, base_payload.get("model")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_TEMPERATURE, base_payload.get("temperature")
    )
    _set_span_attribute(span, SpanAttributes.LLM_TOP_P, base_payload.get("top_p"))
    _set_span_attribute(
        span,
        SpanAttributes.LLM_FREQUENCY_PENALTY,
        base_payload.get("frequency_penalty"),
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_PRESENCE_PENALTY, base_payload.get("presence_penalty")
    )

    _set_span_attribute(
        span, f"{SpanAttributes.LLM_PROMPTS}.0.user", kwargs.get("prompt")
    )

    return


def _set_span_completions(span, llm_request_type, choices):
    if choices is None:
        return

    for index, message in enumerate(choices):
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"

        if llm_request_type == LLMRequestTypeValues.CHAT:
            if message is not None:
                _set_span_attribute(span, f"{prefix}.role", "assistant")
                _set_span_attribute(span, f"{prefix}.content", message)
        elif llm_request_type == LLMRequestTypeValues.COMPLETION:
            _set_span_attribute(span, f"{prefix}.content", message)


def _set_response_attributes(span, llm_request_type, response):
    _set_span_completions(span, llm_request_type, response)

    return


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            # prevent double wrapping
            if hasattr(wrapped, "__wrapped__"):
                return wrapped(*args, **kwargs)

            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def _llm_request_type_by_object(object_name):
    if object_name == "Completion":
        return LLMRequestTypeValues.COMPLETION
    elif object_name == "ChatCompletion":
        return LLMRequestTypeValues.CHAT
    else:
        return LLMRequestTypeValues.UNKNOWN


@_with_tracer_wrapper
def _wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    llm_request_type = _llm_request_type_by_object(to_wrap.get("object"))
    with tracer.start_as_current_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_VENDOR: "OpenAI",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    ) as span:
        try:
            if span.is_recording():
                _set_input_attributes(span, llm_request_type, kwargs)

        except Exception as ex:  # pylint: disable=broad-except
            logger.warning(
                "Failed to set input attributes for openai span, error: %s", str(ex)
            )

        response = wrapped(*args, **kwargs)

        if response:
            try:
                if span.is_recording():
                    _set_response_attributes(span, llm_request_type, response)

            except Exception as ex:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to set response attributes for openai span, error: %s",
                    str(ex),
                )
            if span.is_recording():
                span.set_status(Status(StatusCode.OK))

        return response


class OpenAISpanAttributes:
    OPENAI_API_VERSION = "openai.api_version"
    OPENAI_API_BASE = "openai.api_base"
    OPENAI_API_TYPE = "openai.api_type"


class HaystackInstrumentor(BaseInstrumentor):
    """An instrumentor for the Haystack framework."""

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
                f"{wrap_object}.{wrap_method}" if wrap_object else wrap_method,
                _wrap(tracer, wrapped_method),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            unwrap(
                f"{wrap_package}.{wrap_object}" if wrap_object else wrap_package,
                wrap_method,
            )
