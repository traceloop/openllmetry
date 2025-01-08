"""OpenTelemetry Google Generative AI API instrumentation"""

import logging
import os
import types
from typing import Collection
from opentelemetry.instrumentation.google_generativeai.config import Config
from opentelemetry.instrumentation.google_generativeai.utils import dont_throw
from wrapt import wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, SpanKind
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.trace import Span  # Ensure Span is imported
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap

from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
    LLMRequestTypeValues,
)
from opentelemetry.instrumentation.google_generativeai.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("google-generativeai >= 0.5.0",)

WRAPPED_METHODS = [
    {
        "package": "google.generativeai.generative_models",
        "object": "GenerativeModel",
        "method": "generate_content",
        "span_name": "gemini.generate_content",
    },
    {
        "package": "google.generativeai.generative_models",
        "object": "GenerativeModel",
        "method": "generate_content_async",
        "span_name": "gemini.generate_content_async",
    },
    {
        "package": "google.generativeai.generative_models",
        "object": "ChatSession",
        "method": "send_message",
        "span_name": "gemini.send_message",
    },
]


def should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def is_streaming_response(response):
    return isinstance(response, types.GeneratorType)


def is_async_streaming_response(response):
    return isinstance(response, types.AsyncGeneratorType)


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return



def _emit_prompt_event(span: 'Span', content: str, index: int):
    """Emit a prompt event following the new semantic conventions."""
    attributes = {
        "messaging.role": "user",
        "messaging.content": content,
        "messaging.index": index,
    }
    span.add_event("prompt", attributes=attributes)

def _emit_completion_event(span: Span, content: str, index: int, token_usage: dict = None):
    """Emit a completion event following the new semantic conventions."""
    attributes = {
        "messaging.role": "assistant",
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


def _set_input_attributes(span, args, kwargs, llm_model):
    prompt_content = ""
    if args is not None and len(args) > 0:
        for arg in args:
            if isinstance(arg, str):
                prompt_content = f"{prompt_content}{arg}\n"
            elif isinstance(arg, list):
                for subarg in arg:
                    prompt_content = f"{prompt_content}{subarg}\n"

    if should_send_prompts():
        if Config.use_legacy_attributes:
            # Set legacy prompt attributes if the flag is true
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.0.content",
                prompt_content.strip(),
            )
            _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
        else:
            # Emit prompt event if the flag is false
            _emit_prompt_event(span, prompt_content.strip(), 0)

    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, llm_model)
    if 'prompt' in kwargs and should_send_prompts():
        if Config.use_legacy_attributes:
            # Set legacy prompt attributes if the flag is true
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_PROMPTS}.0.content", kwargs.get("prompt")
            )
            _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
        else:
            # Emit prompt event if the flag is false
            _emit_prompt_event(span, kwargs.get("prompt"), 0)
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TEMPERATURE, kwargs.get("temperature")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_output_tokens")
    )
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TOP_P, kwargs.get("top_p"))
    _set_span_attribute(span, SpanAttributes.LLM_TOP_K, kwargs.get("top_k"))
    _set_span_attribute(
        span, SpanAttributes.LLM_PRESENCE_PENALTY, kwargs.get("presence_penalty")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_FREQUENCY_PENALTY, kwargs.get("frequency_penalty")
    )

    return


@dont_throw
def _set_response_attributes(span, response, llm_model):
    _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, llm_model)

    total_tokens = None
    completion_tokens = None
    prompt_tokens = None

    completions = []

    if hasattr(response, "usage_metadata"):
        total_tokens = response.usage_metadata.total_token_count
        completion_tokens = response.usage_metadata.candidates_token_count
        prompt_tokens = response.usage_metadata.prompt_token_count

        if hasattr(response, 'candidates'):
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            completions.append(part.text)
        elif hasattr(response, 'text'):
            completions.append(response.text)

    else:
        if isinstance(response, list):
            for item in response:
                completions.append(item)
        elif isinstance(response, str):
            completions.append(response)

    if should_send_prompts():
        for index, completion in enumerate(completions):
            if Config.use_legacy_attributes:
                # Set legacy completion attributes if the flag is true
                _set_span_attribute(
                    span, f"{SpanAttributes.LLM_COMPLETIONS}.{index}.content", completion
                )
            else:
                # Emit completion event if the flag is false
                _emit_completion_event(span, completion, index, {
                    "total_tokens": total_tokens,
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens
                })
    else:
        # Neither set legacy attributes nor emit events, only set completion content
        for index, completion in enumerate(completions):
             _set_span_attribute(
                    span, f"{SpanAttributes.LLM_COMPLETIONS}.{index}.content", completion
                )

    if total_tokens is not None:
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            total_tokens,
        )
    if completion_tokens is not None:
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            completion_tokens,
        )
    if prompt_tokens is not None:
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
            prompt_tokens,
        )

    return


def _build_from_streaming_response(span, response, llm_model):
    complete_response = ""
    index = 0
    for item in response:
        item_to_yield = item
        if hasattr(item, 'text'):
            complete_response += str(item.text)
            if not Config.use_legacy_attributes and should_send_prompts():
                # Emit completion event for each chunk in stream if not using legacy attributes
                _emit_completion_event(span, item.text, index)
                index += 1

        yield item_to_yield

    if Config.use_legacy_attributes and should_send_prompts():
        # Set response attributes after streaming is finished if using legacy attributes
        _set_response_attributes(span, complete_response, llm_model)

    span.set_status(Status(StatusCode.OK))
    span.end()



async def _abuild_from_streaming_response(span, response, llm_model):
    complete_response = ""
    index = 0
    async for item in response:
        item_to_yield = item
        if hasattr(item, 'text'):
            complete_response += str(item.text)
            if not Config.use_legacy_attributes and should_send_prompts():
                # Emit completion event for each chunk in stream if not using legacy attributes
                _emit_completion_event(span, item.text, index)
                index += 1

        yield item_to_yield

    if Config.use_legacy_attributes and should_send_prompts():
        # Set response attributes after streaming is finished if using legacy attributes
        _set_response_attributes(span, complete_response, llm_model)

    span.set_status(Status(StatusCode.OK))
    span.end()


@dont_throw
def _handle_request(span, args, kwargs, llm_model):
    if span.is_recording():
        _set_input_attributes(span, args, kwargs, llm_model)


@dont_throw
def _handle_response(span, response, llm_model):
    if span.is_recording():
        _set_response_attributes(span, response, llm_model)

        span.set_status(Status(StatusCode.OK))


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


@_with_tracer_wrapper
async def _awrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return await wrapped(*args, **kwargs)

    llm_model = "unknown"
    if hasattr(instance, "_model_id"):
        llm_model = instance._model_id
    if hasattr(instance, "_model_name"):
        llm_model = instance._model_name.replace("publishers/google/models/", "")

    name = to_wrap.get("span_name")
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Gemini",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
    )

    _handle_request(span, args, kwargs, llm_model)

    response = await wrapped(*args, **kwargs)

    if response:
        if is_streaming_response(response):
            return _build_from_streaming_response(span, response, llm_model)
        elif is_async_streaming_response(response):
            return _abuild_from_streaming_response(span, response, llm_model)
        else:
            _handle_response(span, response, llm_model)

    span.end()
    return response


@_with_tracer_wrapper
def _wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    llm_model = "unknown"
    if hasattr(instance, "_model_id"):
        llm_model = instance._model_id
    if hasattr(instance, "_model_name"):
        llm_model = instance._model_name.replace("publishers/google/models/", "")

    name = to_wrap.get("span_name")
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Gemini",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
    )

    _handle_request(span, args, kwargs, llm_model)

    response = wrapped(*args, **kwargs)

    if response:
        if is_streaming_response(response):
            return _build_from_streaming_response(span, response, llm_model)
        elif is_async_streaming_response(response):
            return _abuild_from_streaming_response(span, response, llm_model)
        else:
            _handle_response(span, response, llm_model)

    span.end()
    return response


class GoogleGenerativeAiInstrumentor(BaseInstrumentor):
    """An instrumentor for Google Generative AI's client library."""

    def __init__(self, exception_logger=None,use_legacy_attributes=True):
        super().__init__()
        Config.exception_logger = exception_logger
        Config.use_legacy_attributes = use_legacy_attributes

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
                (
                    _awrap(tracer, wrapped_method)
                    if wrap_method == "generate_content_async"
                    else _wrap(tracer, wrapped_method)
                ),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"{wrap_package}.{wrap_object}",
                wrapped_method.get("method", ""),
            )
