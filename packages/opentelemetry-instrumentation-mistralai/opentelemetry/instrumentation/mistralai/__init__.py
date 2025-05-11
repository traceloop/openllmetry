"""OpenTelemetry Mistral AI instrumentation"""

import json
import logging
from typing import Collection, Union

from opentelemetry import context as context_api
from opentelemetry._events import get_event_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.mistralai.config import Config
from opentelemetry.instrumentation.mistralai.event_emitter import emit_event
from opentelemetry.instrumentation.mistralai.event_models import (
    ChoiceEvent,
    MessageEvent,
)
from opentelemetry.instrumentation.mistralai.utils import (
    dont_throw,
    should_emit_events,
    should_send_prompts,
)
from opentelemetry.instrumentation.mistralai.version import __version__
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
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

from mistralai.models.chat_completion import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from mistralai.models.common import UsageInfo
from mistralai.models.embeddings import EmbeddingResponse

logger = logging.getLogger(__name__)

_instruments = ("mistralai >= 0.2.0, < 1",)

WRAPPED_METHODS = [
    {
        "method": "chat",
        "span_name": "mistralai.chat",
        "streaming": False,
    },
    {
        "method": "chat_stream",
        "span_name": "mistralai.chat",
        "streaming": True,
    },
    {
        "method": "embeddings",
        "span_name": "mistralai.embeddings",
        "streaming": False,
    },
]


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


@dont_throw
def _set_input_attributes(span, llm_request_type, to_wrap, kwargs):
    if not span.is_recording():
        return
    if should_send_prompts():
        if llm_request_type == LLMRequestTypeValues.CHAT:
            _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
            for index, message in enumerate(kwargs.get("messages")):
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{index}.content",
                    message.content,
                )
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{index}.role",
                    message.role,
                )
        else:
            input = kwargs.get("input")

            if isinstance(input, str):
                _set_span_attribute(
                    span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user"
                )
                _set_span_attribute(
                    span, f"{SpanAttributes.LLM_PROMPTS}.0.content", input
                )
            else:
                for index, prompt in enumerate(input):
                    _set_span_attribute(
                        span,
                        f"{SpanAttributes.LLM_PROMPTS}.{index}.role",
                        "user",
                    )
                    _set_span_attribute(
                        span,
                        f"{SpanAttributes.LLM_PROMPTS}.{index}.content",
                        prompt,
                    )


@dont_throw
def _set_model_input_attributes(span, to_wrap, kwargs):
    if not span.is_recording():
        return
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
    _set_span_attribute(
        span,
        SpanAttributes.LLM_IS_STREAMING,
        to_wrap.get("streaming"),
    )


@dont_throw
def _set_response_attributes(span, llm_request_type, response):
    if llm_request_type == LLMRequestTypeValues.EMBEDDING or not span.is_recording():
        return

    if should_send_prompts():
        for index, choice in enumerate(response.choices):
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
            _set_span_attribute(
                span,
                f"{prefix}.finish_reason",
                choice.finish_reason,
            )
            _set_span_attribute(
                span,
                f"{prefix}.content",
                (
                    choice.message.content
                    if isinstance(choice.message.content, str)
                    else json.dumps(choice.message.content)
                ),
            )
            _set_span_attribute(
                span,
                f"{prefix}.role",
                choice.message.role,
            )


@dont_throw
def _set_model_response_attributes(span, llm_request_type, response):
    if not span.is_recording():
        return

    _set_span_attribute(span, GEN_AI_RESPONSE_ID, response.id)

    if llm_request_type == LLMRequestTypeValues.EMBEDDING:
        return

    _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, response.model)

    if not response.usage:
        return

    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens or 0
    total_tokens = response.usage.total_tokens

    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
        total_tokens,
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


def _accumulate_streaming_response(span, event_logger, llm_request_type, response):
    accumulated_response = ChatCompletionResponse(
        id="",
        object="",
        created=0,
        model="",
        choices=[],
        usage=UsageInfo(prompt_tokens=0, total_tokens=0, completion_tokens=0),
    )

    for res in response:
        yield res

        if res.model:
            accumulated_response.model = res.model
        if res.usage:
            accumulated_response.usage = res.usage
        # Id is the same for all chunks, so it's safe to overwrite it every time
        if res.id:
            accumulated_response.id = res.id

        for idx, choice in enumerate(res.choices):
            if len(accumulated_response.choices) <= idx:
                accumulated_response.choices.append(
                    ChatCompletionResponseChoice(
                        index=idx,
                        message=ChatMessage(role="assistant", content=""),
                        finish_reason=None,
                    )
                )

            accumulated_response.choices[idx].finish_reason = choice.finish_reason
            accumulated_response.choices[idx].message.content += choice.delta.content
            accumulated_response.choices[idx].message.role = choice.delta.role

    _handle_response(span, event_logger, llm_request_type, accumulated_response)

    span.end()


async def _aaccumulate_streaming_response(
    span, event_logger, llm_request_type, response
):
    accumulated_response = ChatCompletionResponse(
        id="",
        object="",
        created=0,
        model="",
        choices=[],
        usage=UsageInfo(prompt_tokens=0, total_tokens=0, completion_tokens=0),
    )

    async for res in response:
        yield res

        if res.model:
            accumulated_response.model = res.model
        if res.usage:
            accumulated_response.usage = res.usage
        # Id is the same for all chunks, so it's safe to overwrite it every time
        if res.id:
            accumulated_response.id = res.id

        for idx, choice in enumerate(res.choices):
            if len(accumulated_response.choices) <= idx:
                accumulated_response.choices.append(
                    ChatCompletionResponseChoice(
                        index=idx,
                        message=ChatMessage(role="assistant", content=""),
                        finish_reason=None,
                    )
                )

            accumulated_response.choices[idx].finish_reason = choice.finish_reason
            accumulated_response.choices[idx].message.content += choice.delta.content
            accumulated_response.choices[idx].message.role = choice.delta.role

    _handle_response(span, event_logger, llm_request_type, accumulated_response)

    span.end()


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, event_logger, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, event_logger, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def _llm_request_type_by_method(method_name):
    if method_name == "chat" or method_name == "chat_stream":
        return LLMRequestTypeValues.CHAT
    elif method_name == "embeddings":
        return LLMRequestTypeValues.EMBEDDING
    else:
        return LLMRequestTypeValues.UNKNOWN


@dont_throw
def _emit_message_events(method_wrapped: str, args, kwargs, event_logger):
    # Handle chat events
    if method_wrapped == "mistralai.chat":
        messages = args[0] if len(args) > 0 else kwargs.get("messages", [])
        for message in messages:
            if isinstance(message, ChatMessage):
                role = message.role
                content = message.content
            elif isinstance(message, dict):
                role = message.get("role", "unknown")
                content = message.get("content")
            emit_event(
                MessageEvent(content=content, role=role or "unknown"), event_logger
            )

    # Handle embedding events
    elif method_wrapped == "mistralai.embeddings":
        embedding_input = args[0] if len(args) > 0 else kwargs.get("input", [])
        if isinstance(embedding_input, str):
            emit_event(MessageEvent(content=embedding_input, role="user"), event_logger)
        elif isinstance(embedding_input, list):
            for prompt in embedding_input:
                emit_event(MessageEvent(content=prompt, role="user"), event_logger)


def _emit_choice_events(
    response: Union[ChatCompletionResponse, EmbeddingResponse], event_logger
):
    # Handle chat events
    if isinstance(response, ChatCompletionResponse):
        for choice in response.choices:
            emit_event(
                ChoiceEvent(
                    index=choice.index,
                    message={
                        "content": choice.message.content,
                        "role": choice.message.role or "assistant",
                    },
                    finish_reason=choice.finish_reason or "unknown",
                ),
                event_logger,
            )

    # Handle embedding events
    elif isinstance(response, EmbeddingResponse):
        for embedding in response.data:
            emit_event(
                ChoiceEvent(
                    index=embedding.index,
                    message={
                        "content": embedding.embedding,
                        "role": "assistant",
                    },
                    finish_reason="unknown",
                ),
                event_logger,
            )


def _handle_input(span, event_logger, args, kwargs, to_wrap):
    name = to_wrap.get("span_name")
    llm_request_type = _llm_request_type_by_method(to_wrap.get("method"))

    _set_model_input_attributes(span, to_wrap, kwargs)

    if should_emit_events() and event_logger:
        _emit_message_events(name, args, kwargs, event_logger)
    else:
        _set_input_attributes(span, llm_request_type, to_wrap, kwargs)


def _handle_response(span, event_logger, llm_request_type, response):
    _set_model_response_attributes(span, llm_request_type, response)

    if should_emit_events() and event_logger:
        _emit_choice_events(response, event_logger)
    else:
        _set_response_attributes(span, llm_request_type, response)


@_with_tracer_wrapper
def _wrap(
    tracer,
    event_logger,
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
            SpanAttributes.LLM_SYSTEM: "MistralAI",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    )

    _handle_input(span, event_logger, args, kwargs, to_wrap)

    response = wrapped(*args, **kwargs)

    if response:
        if to_wrap.get("streaming"):
            return _accumulate_streaming_response(
                span, event_logger, llm_request_type, response
            )

        _handle_response(span, event_logger, llm_request_type, response)

        if span.is_recording():
            span.set_status(Status(StatusCode.OK))

    span.end()
    return response


@_with_tracer_wrapper
async def _awrap(
    tracer,
    event_logger,
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
        return await wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    llm_request_type = _llm_request_type_by_method(to_wrap.get("method"))
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "MistralAI",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    )

    _handle_input(span, event_logger, args, kwargs, to_wrap)

    if to_wrap.get("streaming"):
        response = wrapped(*args, **kwargs)
    else:
        response = await wrapped(*args, **kwargs)

    if response:
        if to_wrap.get("streaming"):
            return _aaccumulate_streaming_response(
                span, event_logger, llm_request_type, response
            )

        _handle_response(span, event_logger, llm_request_type, response)

        if span.is_recording():
            span.set_status(Status(StatusCode.OK))

    span.end()
    return response


class MistralAiInstrumentor(BaseInstrumentor):
    """An instrumentor for Mistral AI's client library."""

    def __init__(self, exception_logger=None, use_legacy_attributes=True):
        super().__init__()
        Config.exception_logger = exception_logger
        Config.use_legacy_attributes = use_legacy_attributes

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        event_logger = None
        if not Config.use_legacy_attributes:
            event_logger_provider = kwargs.get("event_logger_provider")
            event_logger = get_event_logger(
                __name__, __version__, event_logger_provider=event_logger_provider
            )

        for wrapped_method in WRAPPED_METHODS:
            wrap_method = wrapped_method.get("method")
            wrap_function_wrapper(
                "mistralai.client",
                f"MistralClient.{wrap_method}",
                _wrap(tracer, event_logger, wrapped_method),
            )
            wrap_function_wrapper(
                "mistralai.async_client",
                f"MistralAsyncClient.{wrap_method}",
                _awrap(tracer, event_logger, wrapped_method),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            unwrap("mistralai.client.MistralClient", wrapped_method.get("method"))
            unwrap(
                "mistralai.async_client.MistralAsyncClient",
                wrapped_method.get("method"),
            )
