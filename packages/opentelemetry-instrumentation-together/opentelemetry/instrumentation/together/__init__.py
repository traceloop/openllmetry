"""OpenTelemetry Together AI instrumentation"""

import logging
import os
from typing import Collection, Union

from opentelemetry import context as context_api
from opentelemetry._events import Event, EventLogger, get_event_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.together.config import Config
from opentelemetry.instrumentation.together.event_handler import (
    ChoiceEvent,
    MessageEvent,
    emit_event,
)
from opentelemetry.instrumentation.together.utils import dont_throw, is_content_enabled
from opentelemetry.instrumentation.together.version import __version__
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_RESPONSE_ID,
)
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    SpanAttributes,
)
from opentelemetry.trace import SpanKind, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

from together.types.chat_completions import ChatCompletionResponse
from together.types.completions import CompletionResponse

logger = logging.getLogger(__name__)

_instruments = ("together >= 1.2.0, <2",)

WRAPPED_METHODS = [
    {
        "object": "resources",
        "method": "chat.completions.ChatCompletions.create",
        "span_name": "together.chat",
    },
    {
        "object": "resources",
        "method": "completions.Completions.create",
        "span_name": "together.completion",
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
def _set_input_attributes(span, llm_request_type, kwargs):
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
    _set_span_attribute(
        span,
        SpanAttributes.LLM_IS_STREAMING,
        kwargs.get("stream"),
    )

    if should_send_prompts():
        if llm_request_type == LLMRequestTypeValues.CHAT:
            _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
            for index, message in enumerate(kwargs.get("messages")):
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{index}.content",
                    message.get("content"),
                )
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{index}.role",
                    message.get("role"),
                )
        elif llm_request_type == LLMRequestTypeValues.COMPLETION:
            _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_PROMPTS}.0.content", kwargs.get("prompt")
            )


@dont_throw
def _set_response_attributes(span, llm_request_type, response):
    if should_send_prompts():
        if llm_request_type == LLMRequestTypeValues.COMPLETION:
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
                response.choices[0].text,
            )
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant"
            )
        elif llm_request_type == LLMRequestTypeValues.CHAT:
            index = 0
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
            _set_span_attribute(
                span, f"{prefix}.content", response.choices[0].message.content
            )
            _set_span_attribute(
                span, f"{prefix}.role", response.choices[0].message.role
            )

    _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, response.model)
    _set_span_attribute(span, GEN_AI_RESPONSE_ID, response.id)

    usage_data = response.usage
    input_tokens = getattr(usage_data, "prompt_tokens", 0)
    output_tokens = getattr(usage_data, "completion_tokens", 0)

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


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def _llm_request_type_by_method(method_name):
    if method_name == "chat.completions.ChatCompletions.create":
        return LLMRequestTypeValues.CHAT
    elif method_name == "completions.Completions.create":
        return LLMRequestTypeValues.COMPLETION
    else:
        return LLMRequestTypeValues.UNKNOWN


@dont_throw
def _emit_input_events(llm_request_type, kwargs):
    if llm_request_type == LLMRequestTypeValues.CHAT:
        for message in kwargs.get("messages"):
            emit_event(
                MessageEvent(
                    content=message.get("content"), role=message.get("role") or "user"
                )
            )
    elif llm_request_type == LLMRequestTypeValues.COMPLETION:
        emit_event(MessageEvent(content=kwargs.get("prompt"), role="user"))
    else:
        raise ValueError(
            "It wasn't possible to emit the input events due to an unknown llm_request_type."
        )


@dont_throw
def _emit_choice_event(
    llm_request_type, response: Union[ChatCompletionResponse, CompletionResponse]
):
    if llm_request_type == LLMRequestTypeValues.CHAT:
        response: ChatCompletionResponse
        for choice in response.choices:
            emit_event(
                ChoiceEvent(
                    index=choice.index,
                    message={
                        "content": choice.message.content,
                        "role": choice.message.role,
                    },
                    finish_reason=choice.finish_reason,
                )
            )
    elif llm_request_type == LLMRequestTypeValues.COMPLETION:
        response: CompletionResponse
        for choice in response.choices:
            emit_event(
                ChoiceEvent(
                    index=choice.index,
                    message={"content": choice.text, "role": "assistant"},
                    finish_reason=choice.finish_reason,
                )
            )
    else:
        raise ValueError(
            "It wasn't possible to emit the choice events due to an unknown llm_request_type."
        )


@_with_tracer_wrapper
def _wrap(
    tracer,
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
    llm_request_type = _llm_request_type_by_method(to_wrap.get("method"))
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "TogetherAI",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    )
    if span.is_recording():
        _set_input_attributes(span, llm_request_type, kwargs)

    _emit_input_events(llm_request_type, kwargs)

    response = wrapped(*args, **kwargs)

    if response:
        if span.is_recording():
            _set_response_attributes(span, llm_request_type, response)

            _emit_choice_event(llm_request_type, response)

            span.set_status(Status(StatusCode.OK))

    span.end()
    return response


class TogetherAiInstrumentor(BaseInstrumentor):
    """An instrumentor for Together AI's client library."""

    def __init__(self, exception_logger=None, use_legacy_attributes: bool = True):
        super().__init__()
        Config.exception_logger = exception_logger
        Config.use_legacy_attributes = use_legacy_attributes

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        if not Config.use_legacy_attributes:
            event_logger_provider = kwargs.get("event_logger_provider")
            Config.event_logger = get_event_logger(
                __name__, __version__, event_logger_provider=event_logger_provider
            )

        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            wrap_function_wrapper(
                "together",
                f"{wrap_object}.{wrap_method}",
                _wrap(tracer, wrapped_method),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"together.{wrap_object}",
                wrapped_method.get("method"),
            )
