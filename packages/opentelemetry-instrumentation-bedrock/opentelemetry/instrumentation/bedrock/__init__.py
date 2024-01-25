"""OpenTelemetry Bedrock instrumentation"""
from functools import wraps
import json
import logging
import os
from typing import Collection
from opentelemetry.instrumentation.bedrock.reusable_streaming_body import ReusableStreamingBody
from wrapt import wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, SpanKind

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)

from opentelemetry.semconv.ai import SpanAttributes, LLMRequestTypeValues
from opentelemetry.instrumentation.bedrock.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("boto3 >= 1.28.57",)

WRAPPED_METHODS = [
    {
        "package": "botocore.client",
        "object": "ClientCreator",
        "method": "create_client"
    }
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

    if kwargs.get("service_name") == "bedrock-runtime":
        client = wrapped(*args, **kwargs)
        client.invoke_model = _instrumented_model_invoke(client.invoke_model, tracer)

        return client

    return wrapped(*args, **kwargs)


def _instrumented_model_invoke(fn, tracer):
    @wraps(fn)
    def with_instrumentation(*args, **kwargs):
        with tracer.start_as_current_span(
            "bedrock.completion",
            kind=SpanKind.CLIENT
        ) as span:
            response = fn(*args, **kwargs)
            response['body'] = ReusableStreamingBody(response['body']._raw_stream, response['body']._content_length)
            request_body = json.loads(kwargs.get("body"))
            response_body = json.loads(response.get("body").read())

            if span.is_recording():
                (vendor, model) = kwargs.get("modelId").split(".")

                _set_span_attribute(span, SpanAttributes.LLM_VENDOR, vendor)
                _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, model)

                if vendor == "cohere":
                    _set_cohere_span_attributes(span, request_body, response_body)
                elif vendor == "anthropic":
                    _set_anthropic_span_attributes(span, request_body, response_body)
                elif vendor == "ai21":
                    _set_ai21_span_attributes(span, request_body, response_body)
                elif vendor == "meta":
                    _set_llama_span_attributes(span, request_body, response_body)

            return response

    return with_instrumentation


def _set_cohere_span_attributes(span, request_body, response_body):
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.COMPLETION.value)
    _set_span_attribute(span, SpanAttributes.LLM_TOP_P, request_body.get("p"))
    _set_span_attribute(span, SpanAttributes.LLM_TEMPERATURE, request_body.get("temperature"))
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, request_body.get("max_tokens"))

    if should_send_prompts():
        _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.user", request_body.get("prompt"))

        for i, generation in enumerate(response_body.get("generations")):
            _set_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content", generation.get("text"))


def _set_anthropic_span_attributes(span, request_body, response_body):
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.COMPLETION.value)
    _set_span_attribute(span, SpanAttributes.LLM_TOP_P, request_body.get("top_p"))
    _set_span_attribute(span, SpanAttributes.LLM_TEMPERATURE, request_body.get("temperature"))
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, request_body.get("max_tokens_to_sample"))

    if should_send_prompts():
        _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.user", request_body.get("prompt"))
        _set_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", response_body.get("completion"))


def _set_ai21_span_attributes(span, request_body, response_body):
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.COMPLETION.value)
    _set_span_attribute(span, SpanAttributes.LLM_TOP_P, request_body.get("topP"))
    _set_span_attribute(span, SpanAttributes.LLM_TEMPERATURE, request_body.get("temperature"))
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, request_body.get("maxTokens"))

    if should_send_prompts():
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_PROMPTS}.0.user", request_body.get("prompt")
        )

        for i, completion in enumerate(response_body.get("completions")):
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content", completion.get("data").get("text")
            )


def _set_llama_span_attributes(span, request_body, response_body):
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.COMPLETION.value)
    _set_span_attribute(span, SpanAttributes.LLM_TOP_P, request_body.get("top_p"))
    _set_span_attribute(span, SpanAttributes.LLM_TEMPERATURE, request_body.get("temperature"))
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, request_body.get("max_gen_len"))

    if should_send_prompts():
        _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.user", request_body.get("prompt"))

        for i, generation in enumerate(response_body.get("generations")):
            _set_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content", response_body)


class BedrockInstrumentor(BaseInstrumentor):
    """An instrumentor for Bedrock's client library."""

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
                wrapped_method.get("method"),
            )
