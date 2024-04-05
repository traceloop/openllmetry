"""OpenTelemetry Anthropic instrumentation"""

import json
import logging

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

from anthropic._streaming import Stream, AsyncStream

from opentelemetry.semconv.ai import SpanAttributes, LLMRequestTypeValues

from opentelemetry.instrumentation.anthropic.config import Config
from opentelemetry.instrumentation.anthropic.streaming import _build_from_streaming_response, \
    _abuild_from_streaming_response
from opentelemetry.instrumentation.anthropic.utils import _set_span_attribute, should_send_prompts
from opentelemetry.instrumentation.anthropic.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("anthropic >= 0.3.11",)

WRAPPED_METHODS = [
    {
        "package": "anthropic.resources.completions",
        "object": "Completions",
        "method": "create",
        "span_name": "anthropic.completion",
    },
    {
        "package": "anthropic.resources.messages",
        "object": "Messages",
        "method": "create",
        "span_name": "anthropic.completion",
    },
    {
        "package": "anthropic.resources.messages",
        "object": "Messages",
        "method": "stream",
        "span_name": "anthropic.completion",
    },
]
WRAPPED_AMETHODS = [
    {
        "package": "anthropic.resources.completions",
        "object": "AsyncCompletions",
        "method": "create",
        "span_name": "anthropic.completion",
    },
    {
        "package": "anthropic.resources.messages",
        "object": "AsyncMessages",
        "method": "create",
        "span_name": "anthropic.completion",
    },
    {
        "package": "anthropic.resources.messages",
        "object": "AsyncMessages",
        "method": "stream",
        "span_name": "anthropic.completion",
    },
]


def is_streaming_response(response):
    return isinstance(response, Stream) or isinstance(response, AsyncStream)


def _dump_content(content):
    if isinstance(content, str):
        return content
    json_serializable = []
    for item in content:
        if item.get("type") == "text":
            json_serializable.append({"type": "text", "text": item.get("text")})
        elif item.get("type") == "image":
            json_serializable.append(
                {
                    "type": "image",
                    "source": {
                        "type": item.get("source").get("type"),
                        "media_type": item.get("source").get("media_type"),
                        "data": str(item.get("source").get("data")),
                    },
                }
            )
    return json.dumps(json_serializable)


def _set_input_attributes(span, kwargs):
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
    _set_span_attribute(span, SpanAttributes.LLM_IS_STREAMING, kwargs.get("stream"))

    if should_send_prompts():
        if kwargs.get("prompt") is not None:
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_PROMPTS}.0.user", kwargs.get("prompt")
            )

        elif kwargs.get("messages") is not None:
            for i, message in enumerate(kwargs.get("messages")):
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{i}.user",
                    _dump_content(message.get("content")),
                )


def _set_span_completions(span, response):
    index = 0
    prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
    _set_span_attribute(span, f"{prefix}.finish_reason", response.get("stop_reason"))
    if response.get("completion"):
        _set_span_attribute(span, f"{prefix}.content", response.get("completion"))
    elif response.get("content"):
        for i, content in enumerate(response.get("content")):
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content",
                content.text,
            )


async def _set_token_usage_a(span, anthropic, request, response):
    if not isinstance(response, dict):
        response = response.__dict__

    prompt_tokens = 0
    if request.get("prompt"):
        prompt_tokens = await anthropic.count_tokens(request.get("prompt"))
    elif request.get("messages"):
        prompt_tokens = sum(
            [await anthropic.count_tokens(m.get("content")) for m in request.get("messages")]
        )

    completion_tokens = 0
    if response.get("completion"):
        completion_tokens = await anthropic.count_tokens(response.get("completion"))
    elif response.get("content"):
        completion_tokens = await anthropic.count_tokens(response.get("content")[0].text)

    total_tokens = prompt_tokens + completion_tokens

    _set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
    _set_span_attribute(
        span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens
    )
    _set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)


def _set_token_usage(span, anthropic, request, response):
    if not isinstance(response, dict):
        response = response.__dict__

    prompt_tokens = 0
    if request.get("prompt"):
        prompt_tokens = anthropic.count_tokens(request.get("prompt"))
    elif request.get("messages"):
        prompt_tokens = sum(
            [anthropic.count_tokens(m.get("content")) for m in request.get("messages")]
        )

    completion_tokens = 0
    if response.get("completion"):
        completion_tokens = anthropic.count_tokens(response.get("completion"))
    elif response.get("content"):
        completion_tokens = anthropic.count_tokens(response.get("content")[0].text)

    total_tokens = prompt_tokens + completion_tokens

    _set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
    _set_span_attribute(
        span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens
    )
    _set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)


def _set_response_attributes(span, response):
    if not isinstance(response, dict):
        response = response.__dict__
    _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, response.get("model"))

    if response.get("usage"):
        prompt_tokens = response.get("usage").input_tokens
        completion_tokens = response.get("usage").output_tokens
        _set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
        _set_span_attribute(
            span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            prompt_tokens + completion_tokens,
        )

    if should_send_prompts():
        _set_span_completions(span, response)


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
            SpanAttributes.LLM_VENDOR: "Anthropic",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
    )
    try:
        if span.is_recording():
            _set_input_attributes(span, kwargs)

    except Exception as ex:  # pylint: disable=broad-except
        logger.warning(
            "Failed to set input attributes for anthropic span, error: %s", str(ex)
        )

    response = wrapped(*args, **kwargs)

    if is_streaming_response(response):
        return _build_from_streaming_response(span, response, instance._client, kwargs)
    elif response:
        try:
            if span.is_recording():
                _set_response_attributes(span, response)
                _set_token_usage(span, instance._client, kwargs, response)

        except Exception as ex:  # pylint: disable=broad-except
            logger.warning(
                "Failed to set response attributes for anthropic span, error: %s",
                str(ex),
            )
        if span.is_recording():
            span.set_status(Status(StatusCode.OK))
    span.end()
    return response


@_with_tracer_wrapper
async def _awrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_VENDOR: "Anthropic",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
    )
    try:
        if span.is_recording():
            _set_input_attributes(span, kwargs)

    except Exception as ex:  # pylint: disable=broad-except
        logger.warning(
            "Failed to set input attributes for anthropic span, error: %s", str(ex)
        )

    response = await wrapped(*args, **kwargs)

    if is_streaming_response(response):
        return _abuild_from_streaming_response(span, response, instance._client, kwargs)
    elif response:
        try:
            if span.is_recording():
                _set_response_attributes(span, response)
                await _set_token_usage_a(span, instance._client, kwargs, response)

        except Exception as ex:  # pylint: disable=broad-except
            logger.warning(
                "Failed to set response attributes for anthropic span, error: %s",
                str(ex),
            )
        if span.is_recording():
            span.set_status(Status(StatusCode.OK))
    span.end()
    return response


class AnthropicInstrumentor(BaseInstrumentor):
    """An instrumentor for Anthropic's client library."""

    def __init__(
            self, enrich_token_usage: bool = False
    ):
        super().__init__()
        Config.enrich_token_usage = enrich_token_usage

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
        for wrapped_method in WRAPPED_AMETHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            wrap_function_wrapper(
                wrap_package,
                f"{wrap_object}.{wrap_method}",
                _awrap(tracer, wrapped_method),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"anthropic.resources.completions.{wrap_object}",
                wrapped_method.get("method"),
            )
        for wrapped_method in WRAPPED_AMETHODS:
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"anthropic.resources.completions.{wrap_object}",
                wrapped_method.get("method"),
            )
