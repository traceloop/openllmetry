"""OpenTelemetry Cohere instrumentation"""

import logging
import os
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
from opentelemetry.instrumentation.cohere.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("cohere >= 4.2.7",)

WRAPPED_METHODS = [
    {
        "object": "Client",
        "method": "generate",
        "span_name": "cohere.completion",
    },
    {
        "object": "Client",
        "method": "chat",
        "span_name": "cohere.chat",
    },
    {
        "object": "Client",
        "method": "rerank",
        "span_name": "cohere.rerank",
    },
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


def _set_input_attributes(span, llm_request_type, kwargs):
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_tokens_to_sample")
    )
    _set_span_attribute(span, SpanAttributes.LLM_TEMPERATURE, kwargs.get("temperature"))
    _set_span_attribute(span, SpanAttributes.LLM_TOP_P, kwargs.get("top_p"))
    _set_span_attribute(
        span, SpanAttributes.LLM_FREQUENCY_PENALTY, kwargs.get("frequency_penalty")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_PRESENCE_PENALTY, kwargs.get("presence_penalty")
    )

    if should_send_prompts():
        if llm_request_type == LLMRequestTypeValues.COMPLETION:
            _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_PROMPTS}.0.content", kwargs.get("prompt")
            )
        elif llm_request_type == LLMRequestTypeValues.CHAT:
            _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_PROMPTS}.0.content", kwargs.get("message")
            )
        elif llm_request_type == LLMRequestTypeValues.RERANK:
            for index, document in enumerate(kwargs.get("documents")):
                _set_span_attribute(
                    span, f"{SpanAttributes.LLM_PROMPTS}.{index}.role", "system"
                )
                _set_span_attribute(
                    span, f"{SpanAttributes.LLM_PROMPTS}.{index}.content", document
                )

            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.{len(kwargs.get('documents'))}.role",
                "user",
            )
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.{len(kwargs.get('documents'))}.content",
                kwargs.get("query"),
            )

    return


def _set_span_chat_response(span, response):
    index = 0
    prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
    _set_span_attribute(span, f"{prefix}.content", response.text)

    if response.token_count:
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            response.token_count.get("total_tokens"),
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            response.token_count.get("response_tokens"),
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
            response.token_count.get("prompt_tokens"),
        )


def _set_span_generations_response(span, generations):
    for index, generation in enumerate(generations):
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
        _set_span_attribute(span, f"{prefix}.content", generation.text)


def _set_span_rerank_response(span, response):
    for idx, doc in enumerate(response.results):
        print(doc.index, doc.relevance_score, doc.document["text"])
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{idx}"
        _set_span_attribute(span, f"{prefix}.role", "assistant")
        content = f"Doc {doc.index}, Score: {doc.relevance_score}"
        if doc.document["text"]:
            content += f"\n{doc.document['text']}"
        _set_span_attribute(
            span,
            f"{prefix}.content",
            content,
        )


def _set_response_attributes(span, llm_request_type, response):
    if should_send_prompts():
        if llm_request_type == LLMRequestTypeValues.CHAT:
            _set_span_chat_response(span, response)
        elif llm_request_type == LLMRequestTypeValues.COMPLETION:
            _set_span_generations_response(span, response)
        elif llm_request_type == LLMRequestTypeValues.RERANK:
            _set_span_rerank_response(span, response)


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def _llm_request_type_by_method(method_name):
    if method_name == "chat":
        return LLMRequestTypeValues.CHAT
    elif method_name == "generate":
        return LLMRequestTypeValues.COMPLETION
    elif method_name == "rerank":
        return LLMRequestTypeValues.RERANK
    else:
        return LLMRequestTypeValues.UNKNOWN


@_with_tracer_wrapper
def _wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    llm_request_type = _llm_request_type_by_method(to_wrap.get("method"))
    with tracer.start_as_current_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_VENDOR: "Cohere",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    ) as span:
        try:
            if span.is_recording():
                _set_input_attributes(span, llm_request_type, kwargs)

        except Exception as ex:  # pylint: disable=broad-except
            logger.warning(
                "Failed to set input attributes for cohere span, error: %s", str(ex)
            )

        response = wrapped(*args, **kwargs)

        if response:
            try:
                if span.is_recording():
                    _set_response_attributes(span, llm_request_type, response)

            except Exception as ex:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to set response attributes for cohere span, error: %s",
                    str(ex),
                )
            if span.is_recording():
                span.set_status(Status(StatusCode.OK))

        return response


class CohereInstrumentor(BaseInstrumentor):
    """An instrumentor for Cohere's client library."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            wrap_function_wrapper(
                "cohere.client",
                f"{wrap_object}.{wrap_method}",
                _wrap(tracer, wrapped_method),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"cohere.client.{wrap_object}",
                wrapped_method.get("method"),
            )
