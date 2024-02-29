"""OpenTelemetry IBM Watsonx AI instrumentation"""

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
from opentelemetry.instrumentation.watsonx.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("ibm_watson_machine_learning >= 1.0.347",)

WRAPPED_METHODS_WATSON_ML_VERSION_1 = [
    {
        "module": "ibm_watson_machine_learning.foundation_models.inference",
        "object": "ModelInference",
        "method": "__init__",
        "span_name": "watsonx.model_init",
    },
    {
        "module": "ibm_watson_machine_learning.foundation_models.inference",
        "object": "ModelInference",
        "method": "generate",
        "span_name": "watsonx.generate",
    },
    {
        "module": "ibm_watson_machine_learning.foundation_models.inference",
        "object": "ModelInference",
        "method": "generate_text_stream",
        "span_name": "watsonx.generate_text_stream",
    },
    {
        "module": "ibm_watson_machine_learning.foundation_models.inference",
        "object": "ModelInference",
        "method": "get_details",
        "span_name": "watsonx.get_details",
    },
]

WRAPPED_METHODS_WATSON_AI_VERSION_1 = [
    {
        "module": "ibm_watsonx_ai.foundation_models",
        "object": "ModelInference",
        "method": "__init__",
        "span_name": "watsonx.model_init",
    },
    {
        "module": "ibm_watsonx_ai.foundation_models",
        "object": "ModelInference",
        "method": "generate",
        "span_name": "watsonx.generate",
    },
    {
        "module": "ibm_watsonx_ai.foundation_models",
        "object": "ModelInference",
        "method": "generate_text_stream",
        "span_name": "watsonx.generate_text_stream",
    },
    {
        "module": "ibm_watsonx_ai.foundation_models",
        "object": "ModelInference",
        "method": "get_details",
        "span_name": "watsonx.get_details",
    },
]

WATSON_MODULES = [
    WRAPPED_METHODS_WATSON_ML_VERSION_1,
    WRAPPED_METHODS_WATSON_AI_VERSION_1,
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
        if isinstance(prompt, list):
            for index, input in enumerate(prompt):
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{index}.user",
                    input,
                )
        else:
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.user",
                prompt,
            )

    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, instance.model_id)
    # Set other attributes
    modelParameters = instance.params
    if modelParameters is not None:
        _set_span_attribute(
            span, SpanAttributes.LLM_DECODING_METHOD, modelParameters.get("decoding_method", None)
        )
        _set_span_attribute(
            span, SpanAttributes.LLM_RANDOM_SEED, modelParameters.get("random_seed", None)
        )
        _set_span_attribute(
            span, SpanAttributes.LLM_MAX_NEW_TOKENS, modelParameters.get("max_new_tokens", None)
        )
        _set_span_attribute(
            span, SpanAttributes.LLM_MIN_NEW_TOKENS, modelParameters.get("min_new_tokens", None)
        )
        _set_span_attribute(span, SpanAttributes.LLM_TOP_K, modelParameters.get("top_k", None))
        _set_span_attribute(
            span,
            SpanAttributes.LLM_REPETITION_PENALTY,
            modelParameters.get("repetition_penalty", None),
        )
        _set_span_attribute(
            span, SpanAttributes.LLM_TEMPERATURE, modelParameters.get("temperature", None)
        )
        _set_span_attribute(span, SpanAttributes.LLM_TOP_P, modelParameters.get("top_p", None))

    return


def _set_stream_response_attributes(span, stream_response):
    _set_span_attribute(
        span,
        SpanAttributes.LLM_RESPONSE_MODEL,
        stream_response.get("model_id")
    )
    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
        stream_response.get("input_token_count"),
    )
    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
        stream_response.get("generated_token_count"),
    )
    total_token = stream_response.get("input_token_count") + stream_response.get("generated_token_count")
    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
        total_token,
        )
    _set_span_attribute(
        span,
        f"{SpanAttributes.LLM_COMPLETIONS}.content",
        stream_response.get("generated_text"),
    )


def _set_single_response_attributes(span, response, index) -> int:

    token_sum = 0
    if not isinstance(response, (dict, str)):
        return token_sum

    if isinstance(response, str):
        if index is not None:
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_COMPLETIONS}.{index}.content",
                response,
            )
        else:
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_COMPLETIONS}.content",
                response,
            )
        return token_sum

    usage = response["results"][0]
    token_sum = usage.get("input_token_count") + usage.get("generated_token_count")
    _set_span_attribute(
        span, SpanAttributes.LLM_RESPONSE_MODEL, response.get("model_id")
    )

    if index is not None:
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_USAGE_COMPLETION_TOKENS}.{index}",
            usage.get("generated_token_count"),
        )
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_USAGE_PROMPT_TOKENS}.{index}",
            usage.get("input_token_count"),
        )
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_COMPLETIONS}.{index}.content",
            response["results"][0]["generated_text"],
        )
    else:
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_USAGE_COMPLETION_TOKENS}",
            usage.get("generated_token_count"),
        )
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_USAGE_PROMPT_TOKENS}",
            usage.get("input_token_count"),
        )
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_COMPLETIONS}.content",
            response["results"][0]["generated_text"],
        )

    return token_sum


def _set_response_attributes(span, responses):
    total_token = 0
    if isinstance(responses, list) and len(responses) > 1:
        for index, response in enumerate(responses):
            total_token += _set_single_response_attributes(span, response, index)
    elif isinstance(responses, list) and len(responses) == 1:
        response = responses[0]
        total_token = _set_single_response_attributes(span, response, None)
    elif isinstance(responses, dict):
        response = responses
        total_token = _set_single_response_attributes(span, response, None)

    if total_token != 0:
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_USAGE_TOTAL_TOKENS}",
            total_token,
            )

    return


def _build_and_set_stream_response(span, response, raw_flag):
    stream_generated_text = ""
    stream_generated_token_count = 0
    stream_input_token_count = 0
    for item in response:
        stream_model_id = item["model_id"]
        stream_generated_text += item["results"][0]["generated_text"]
        stream_input_token_count += item["results"][0]["input_token_count"]
        stream_generated_token_count = item["results"][0]["generated_token_count"]

        if raw_flag:
            yield item
        else:
            yield item["results"][0]["generated_text"]

    stream_response = {
        "model_id": stream_model_id,
        "generated_text": stream_generated_text,
        "generated_token_count": stream_generated_token_count,
        "input_token_count": stream_input_token_count,
    }
    _set_stream_response_attributes(span, stream_response)

    span.set_status(Status(StatusCode.OK))
    span.end()


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
    if "model_init" not in name:
        _set_input_attributes(span, instance, kwargs)
        if to_wrap.get("method") == "generate_text_stream":
            if (raw_flag := kwargs.get("raw_response", None)) is None:
                kwargs = {**kwargs, "raw_response": True}
            elif raw_flag is False:
                kwargs["raw_response"] = True

    response = wrapped(*args, **kwargs)

    if "model_init" not in name:
        if isinstance(response, types.GeneratorType):
            return _build_and_set_stream_response(span, response, raw_flag)
        else:
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

        for wrapped_methods in WATSON_MODULES:
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
        for wrapped_methods in WATSON_MODULES:
            for wrapped_method in wrapped_methods:
                wrap_module = wrapped_method.get("module")
                wrap_object = wrapped_method.get("object")
                unwrap(f"{wrap_module}.{wrap_object}", wrapped_method.get("method"))
