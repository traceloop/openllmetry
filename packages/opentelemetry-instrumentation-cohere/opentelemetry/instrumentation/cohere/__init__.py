"""OpenTelemetry Cohere instrumentation"""

import logging
import os
from typing import Collection
from opentelemetry.instrumentation.cohere.config import Config
from opentelemetry.instrumentation.cohere.utils import dont_throw
from wrapt import wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, SpanKind
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)

from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
    LLMRequestTypeValues,
)
from opentelemetry.instrumentation.cohere.version import __version__
from opentelemetry.trace.span import Span
from opentelemetry.util.types import Attributes


logger = logging.getLogger(__name__)

_instruments = ("cohere >=4.2.7, <6",)

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

def _emit_prompt_event(span: Span, role: str, content: str, index: int):
    """Emit a prompt event following the new semantic conventions."""
    attributes: Attributes = {
        "messaging.role": role,
        "messaging.content": content,
        "messaging.index": index,
    }
    span.add_event("prompt", attributes=attributes)

def _emit_completion_event(span: Span, content: str, index: int, token_usage: dict = None):
    """Emit a completion event following the new semantic conventions."""
    attributes: Attributes = {
        "messaging.content": content,
        "messaging.index": index,
    }
    if token_usage:
        attributes.update({
            "llm.usage.total_tokens": token_usage.get("total_tokens"),
            "llm.usage.prompt_tokens": token_usage.get("prompt_tokens"),
            "llm.usage.completion_tokens": token_usage.get("completion_tokens"),
        })
    span.add_event("completion", attributes=attributes)


@dont_throw
def _set_input_attributes(span, llm_request_type, kwargs):
    # Always set these basic attributes regardless of configuration
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_tokens_to_sample"))
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TEMPERATURE, kwargs.get("temperature"))
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TOP_P, kwargs.get("top_p"))
    _set_span_attribute(span, SpanAttributes.LLM_FREQUENCY_PENALTY, kwargs.get("frequency_penalty"))
    _set_span_attribute(span, SpanAttributes.LLM_PRESENCE_PENALTY, kwargs.get("presence_penalty"))

    if should_send_prompts():
        if Config.use_legacy_attributes:
            # Legacy attribute-based approach
            if llm_request_type == LLMRequestTypeValues.COMPLETION:
                _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
                _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content", kwargs.get("prompt"))
            elif llm_request_type == LLMRequestTypeValues.CHAT:
                _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
                _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content", kwargs.get("message"))
            elif llm_request_type == LLMRequestTypeValues.RERANK:
                for index, document in enumerate(kwargs.get("documents", [])):
                    _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.{index}.role", "system")
                    _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.{index}.content", document)
                _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.{len(kwargs.get('documents'))}.role", "user")
                _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.{len(kwargs.get('documents'))}.content", kwargs.get("query"))
        else:
            # New event-based approach
            if llm_request_type == LLMRequestTypeValues.COMPLETION:
                _emit_prompt_event(span, "user", kwargs.get("prompt"), 0)
            elif llm_request_type == LLMRequestTypeValues.CHAT:
                _emit_prompt_event(span, "user", kwargs.get("message"), 0)
            elif llm_request_type == LLMRequestTypeValues.RERANK:
                for index, document in enumerate(kwargs.get("documents", [])):
                    _emit_prompt_event(span, "system", document, index)
                _emit_prompt_event(span, "user", kwargs.get("query"), len(kwargs.get("documents", [])))


def _set_span_chat_response(span, response):
    index = 0
    prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
    _set_span_attribute(span, f"{prefix}.content", response.text)

    # Cohere v4
    if hasattr(response, "token_count"):
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

    # Cohere v5
    if hasattr(response, "meta") and hasattr(response.meta, "billed_units"):
        input_tokens = response.meta.billed_units.input_tokens
        output_tokens = response.meta.billed_units.output_tokens

        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            input_tokens + output_tokens,
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            output_tokens,
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
            input_tokens,
        )


def _set_span_generations_response(span, response):
    if hasattr(response, "generations"):
        generations = response.generations  # Cohere v5
    else:
        generations = response  # Cohere v4

    for index, generation in enumerate(generations):
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
        _set_span_attribute(span, f"{prefix}.content", generation.text)


def _set_span_rerank_response(span, response):
    for idx, doc in enumerate(response.results):
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{idx}"
        _set_span_attribute(span, f"{prefix}.role", "assistant")
        content = f"Doc {doc.index}, Score: {doc.relevance_score}"
        if doc.document:
            if hasattr(doc.document, "text"):
                content += f"\n{doc.document.text}"
            else:
                content += f"\n{doc.document.get('text')}"
        _set_span_attribute(
            span,
            f"{prefix}.content",
            content,
        )


@dont_throw
def _set_response_attributes(span, llm_request_type, response):
    """Set response attributes using either legacy or new event-based approach."""
    if not should_send_prompts():
        return
    
    if Config.use_legacy_attributes:
        if llm_request_type == LLMRequestTypeValues.CHAT:
            _set_span_chat_response(span, response)
        elif llm_request_type == LLMRequestTypeValues.COMPLETION:
            _set_span_generations_response(span, response)
        elif llm_request_type == LLMRequestTypeValues.RERANK:
            _set_span_rerank_response(span, response)

    else:
        if llm_request_type == LLMRequestTypeValues.CHAT:
            token_usage = None
            if hasattr(response, "token_count"):
                token_usage = {
                    "total_tokens": response.token_count.get("total_tokens"),
                    "prompt_tokens": response.token_count.get("prompt_tokens"),
                    "completion_tokens": response.token_count.get("response_tokens")
                }
            elif hasattr(response, "meta") and hasattr(response.meta, "billed_units"):
                token_usage = {
                    "total_tokens": response.meta.billed_units.input_tokens + response.meta.billed_units.output_tokens,
                    "prompt_tokens": response.meta.billed_units.input_tokens,
                    "completion_tokens": response.meta.billed_units.output_tokens
                }
            _emit_completion_event(span, response.text, 0, token_usage)

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
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    llm_request_type = _llm_request_type_by_method(to_wrap.get("method"))
    with tracer.start_as_current_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Cohere",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    ) as span:
        if span.is_recording():
            _set_input_attributes(span, llm_request_type, kwargs)

        response = wrapped(*args, **kwargs)

        if response:
            if span.is_recording():
                _set_response_attributes(span, llm_request_type, response)
                span.set_status(Status(StatusCode.OK))

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
