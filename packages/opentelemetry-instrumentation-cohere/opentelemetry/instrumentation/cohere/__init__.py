"""OpenTelemetry Cohere instrumentation"""

import logging
import os
from typing import Collection
from opentelemetry.instrumentation.cohere.config import Config
from opentelemetry.instrumentation.cohere.events import (
    create_prompt_event,
    create_completion_event,
    create_rerank_event,
)
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


@dont_throw
def _set_input_attributes(span, llm_request_type, kwargs, event_logger=None, use_legacy_attributes=True):
    model = kwargs.get("model")

    if not use_legacy_attributes and event_logger:
        if llm_request_type == LLMRequestTypeValues.COMPLETION:
            event_logger.add_event(
                create_prompt_event(
                    content=kwargs.get("prompt"),
                    role="user",
                    model=model,
                )
            )
        elif llm_request_type == LLMRequestTypeValues.CHAT:
            event_logger.add_event(
                create_prompt_event(
                    content=kwargs.get("message"),
                    role="user",
                    model=model,
                )
            )
        elif llm_request_type == LLMRequestTypeValues.RERANK:
            events = create_rerank_event(
                documents=kwargs.get("documents"),
                query=kwargs.get("query"),
                model=model,
            )
            for event in events:
                event_logger.add_event(event)
    else:
        _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, model)
        _set_span_attribute(
            span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_tokens_to_sample")
        )
        _set_span_attribute(
            span, SpanAttributes.LLM_REQUEST_TEMPERATURE, kwargs.get("temperature")
        )
        _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TOP_P, kwargs.get("top_p"))
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


def _set_span_chat_response(span, response, event_logger=None, use_legacy_attributes=True, model=None):
    if not use_legacy_attributes and event_logger:
        # Get token counts
        completion_tokens = None
        if hasattr(response, "token_count"):  # Cohere v4
            completion_tokens = response.token_count.get("response_tokens")
        elif hasattr(response, "meta") and hasattr(response.meta, "billed_units"):  # Cohere v5
            completion_tokens = response.meta.billed_units.output_tokens

        event_logger.add_event(
            create_completion_event(
                completion=response.text,
                model=model,
                completion_tokens=completion_tokens,
                role="assistant",
            )
        )
        return

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


def _set_span_generations_response(span, response, event_logger=None, use_legacy_attributes=True, model=None):
    if hasattr(response, "generations"):
        generations = response.generations  # Cohere v5
    else:
        generations = response  # Cohere v4

    if not use_legacy_attributes and event_logger:
        for generation in generations:
            event_logger.add_event(
                create_completion_event(
                    completion=generation.text,
                    model=model,
                )
            )
        return

    for index, generation in enumerate(generations):
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
        _set_span_attribute(span, f"{prefix}.content", generation.text)


def _set_span_rerank_response(span, response, event_logger=None, use_legacy_attributes=True, model=None):
    if not use_legacy_attributes and event_logger:
        scores = [doc.relevance_score for doc in response.results]
        indices = [doc.index for doc in response.results]
        documents = []
        for doc in response.results:
            if doc.document:
                if hasattr(doc.document, "text"):
                    documents.append(doc.document.text)
                else:
                    documents.append(doc.document.get("text"))
            else:
                documents.append("")

        events = create_rerank_event(
            documents=documents,
            query="",  # Query already logged in input
            model=model,
            scores=scores,
            indices=indices,
        )
        for event in events[len(documents)+1:]:  # Skip prompt events
            event_logger.add_event(event)
        return

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
def _set_response_attributes(span, llm_request_type, response, event_logger=None, use_legacy_attributes=True, model=None):
    if should_send_prompts():
        if llm_request_type == LLMRequestTypeValues.CHAT:
            _set_span_chat_response(span, response, event_logger, use_legacy_attributes, model)
        elif llm_request_type == LLMRequestTypeValues.COMPLETION:
            _set_span_generations_response(span, response, event_logger, use_legacy_attributes, model)
        elif llm_request_type == LLMRequestTypeValues.RERANK:
            _set_span_rerank_response(span, response, event_logger, use_legacy_attributes, model)


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
            _set_input_attributes(
                span, 
                llm_request_type, 
                kwargs, 
                event_logger=instance._event_logger if hasattr(instance, "_event_logger") else None,
                use_legacy_attributes=instance._use_legacy_attributes if hasattr(instance, "_use_legacy_attributes") else True,
            )

        response = wrapped(*args, **kwargs)

        if response:
            if span.is_recording():
                _set_response_attributes(
                    span, 
                    llm_request_type, 
                    response,
                    event_logger=instance._event_logger if hasattr(instance, "_event_logger") else None,
                    use_legacy_attributes=instance._use_legacy_attributes if hasattr(instance, "_use_legacy_attributes") else True,
                    model=kwargs.get("model"),
                )
                span.set_status(Status(StatusCode.OK))

        return response


class CohereInstrumentor(BaseInstrumentor):
    """An instrumentor for Cohere's client library."""

    def __init__(self, exception_logger=None, use_legacy_attributes=True):
        super().__init__()
        self.exception_logger = exception_logger
        self.use_legacy_attributes = use_legacy_attributes

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        event_logger = kwargs.get("event_logger")

        for wrapped_method in WRAPPED_METHODS:
            wrap_function_wrapper(
                "cohere",
                f"Client.{wrapped_method.get('method')}",
                _wrap(tracer, wrapped_method),
            )

            # Patch the Client class to include event logger and config
            from cohere import Client
            Client._event_logger = event_logger
            Client._use_legacy_attributes = self.use_legacy_attributes

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            unwrap(
                "cohere.Client",
                wrapped_method.get("method"),
            )

        # Remove patched attributes
        from cohere import Client
        if hasattr(Client, "_event_logger"):
            delattr(Client, "_event_logger")
        if hasattr(Client, "_use_legacy_attributes"):
            delattr(Client, "_use_legacy_attributes")
