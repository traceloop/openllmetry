"""OpenTelemetry IBM Watsonx AI instrumentation"""

import logging
import os
from typing import Collection
from wrapt import wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, SpanKind

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)

from opentelemetry.semconv.ai import SpanAttributes, LLMRequestTypeValues
from opentelemetry.instrumentation.watsonx.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("ibm_watson_machine_learning >= 1.0.327",)

WRAPPED_METHODS_VERSION_1 = [
    {
        "module": "ibm_watson_machine_learning.foundation_models",
        "object": "Model",
        "method": "generate",
        "span_name": "watsonx.generate",
    },
    {
        "module": "ibm_watson_machine_learning.foundation_models",
        "object": "Model",
        "method": "generate_text",
        "span_name": "watsonx.generate_text",
    },
    {
        "module": "ibm_watson_machine_learning.foundation_models",
        "object": "Model",
        "method": "generate_text_stream",
        "span_name": "watsonx.generate_text_stream",
    },
    {
        "module": "ibm_watson_machine_learning.foundation_models",
        "object": "Model",
        "method": "get_details",
        "span_name": "watsonx.get_details",
    },
]


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _set_api_attributes(span):
    _set_span_attribute(
        span,
        WatsonxSpanAttributes.WATSONX_API_BASE,
        "https://us-south.ml.cloud.ibm.com",
    )
    _set_span_attribute(span, WatsonxSpanAttributes.WATSONX_API_TYPE, "watsonx.ai")
    _set_span_attribute(span, WatsonxSpanAttributes.WATSONX_API_VERSION, "1.0")

    return


def should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def _set_input_attributes(span, instance, kwargs):
    if should_send_prompts() and kwargs is not None and len(kwargs) > 0:
        prompt = kwargs.get("prompt")
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_PROMPTS}.0.user",
            prompt,
        )

    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, instance.model_id)
    # Set other attributes
    modelParameters = instance.params
    _set_span_attribute(
        span, SpanAttributes.LLM_DECODING_METHOD, modelParameters.get("decoding_method")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_RANDOM_SEED, modelParameters.get("random_seed")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_MAX_NEW_TOKENS, modelParameters.get("max_new_tokens")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_MIN_NEW_TOKENS, modelParameters.get("min_new_tokens")
    )
    _set_span_attribute(span, SpanAttributes.LLM_TOP_K, modelParameters.get("top_k"))
    _set_span_attribute(
        span,
        SpanAttributes.LLM_REPETITION_PENALTY,
        modelParameters.get("repetition_penalty"),
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_TEMPERATURE, modelParameters.get("temperature")
    )
    _set_span_attribute(span, SpanAttributes.LLM_TOP_P, modelParameters.get("top_p"))

    return


def _set_response_attributes(span, response):
    _set_span_attribute(
        span, SpanAttributes.LLM_RESPONSE_MODEL, response.get("model_id")
    )

    usage = response["results"][0]
    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
        usage.get("input_token_count") + usage.get("generated_token_count"),
    )
    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
        usage.get("generated_token_count"),
    )
    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
        usage.get("input_token_count"),
    )
    _set_span_attribute(
        span,
        f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
        response["results"][0]["generated_text"],
    )

    return


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
            SpanAttributes.LLM_VENDOR: "Watsonx",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
    )

    _set_api_attributes(span)
    _set_input_attributes(span, instance, kwargs)

    response = wrapped(*args, **kwargs)

    _set_response_attributes(span, response)

    span.end()
    return response


class WatsonxSpanAttributes:
    WATSONX_API_VERSION = "watsonx.api_version"
    WATSONX_API_BASE = "watsonx.api_base"
    WATSONX_API_TYPE = "watsonx.api_type"


class WatsonxInstrumentor(BaseInstrumentor):
    """An instrumentor for Watsonx's client library."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        wrapped_methods = WRAPPED_METHODS_VERSION_1
        for wrapped_method in wrapped_methods:
            wrap_module = wrapped_method.get("module")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            wrap_function_wrapper(
                wrap_module,
                f"{wrap_object}.{wrap_method}",
                _wrap(tracer, wrapped_method),
            )

    def _uninstrument(self, **kwargs):
        wrapped_methods = WRAPPED_METHODS_VERSION_1
        for wrapped_method in wrapped_methods:
            wrap_object = wrapped_method.get("object")
            unwrap(f"openai.{wrap_object}", wrapped_method.get("method"))
