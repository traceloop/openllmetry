import logging
import os
import types
import pkg_resources
from typing import Collection
from wrapt import wrap_function_wrapper
import openai

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, SpanKind
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)

from opentelemetry.semconv.ai import SpanAttributes, LLMRequestTypeValues
from opentelemetry.instrumentation.openai.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("openai >= 0.27.0",)

WRAPPED_METHODS_VERSION_0 = [
    {
        "module": "openai",
        "object": "ChatCompletion",
        "method": "create",
        "span_name": "openai.chat",
    },
    {
        "module": "openai",
        "object": "Completion",
        "method": "create",
        "span_name": "openai.completion",
    },
]

WRAPPED_METHODS_VERSION_1 = [
    {
        "module": "openai.resources.chat.completions",
        "object": "Completions",
        "method": "create",
        "span_name": "openai.chat",
    },
    {
        "module": "openai.resources.completions",
        "object": "Completions",
        "method": "create",
        "span_name": "openai.completion",
    },
]


def should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def is_openai_v1():
    return pkg_resources.get_distribution("openai").version >= "1.0.0"


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _set_api_attributes(span):
    _set_span_attribute(
        span,
        OpenAISpanAttributes.OPENAI_API_BASE,
        openai.base_url if hasattr(openai, "base_url") else openai.api_base,
    )
    _set_span_attribute(span, OpenAISpanAttributes.OPENAI_API_TYPE, openai.api_type)
    _set_span_attribute(
        span, OpenAISpanAttributes.OPENAI_API_VERSION, openai.api_version
    )

    return


def _set_span_prompts(span, messages):
    if messages is None:
        return

    for i, msg in enumerate(messages):
        prefix = f"{SpanAttributes.LLM_PROMPTS}.{i}"
        _set_span_attribute(span, f"{prefix}.role", msg.get("role"))
        _set_span_attribute(span, f"{prefix}.content", msg.get("content"))


def _set_input_attributes(span, llm_request_type, kwargs):
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_tokens")
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
        if llm_request_type == LLMRequestTypeValues.CHAT:
            _set_span_prompts(span, kwargs.get("messages"))
        elif llm_request_type == LLMRequestTypeValues.COMPLETION:
            prompt = kwargs.get("prompt")
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.0.user",
                prompt[0] if isinstance(prompt, list) else prompt,
            )

    return


def _set_span_completions(span, llm_request_type, choices):
    if choices is None:
        return

    for choice in choices:
        if is_openai_v1() and not isinstance(choice, dict):
            choice = choice.__dict__

        index = choice.get("index")
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
        _set_span_attribute(
            span, f"{prefix}.finish_reason", choice.get("finish_reason")
        )

        if llm_request_type == LLMRequestTypeValues.CHAT:
            message = choice.get("message")
            if message is not None:
                if is_openai_v1() and not isinstance(message, dict):
                    message = message.__dict__

                _set_span_attribute(span, f"{prefix}.role", message.get("role"))
                _set_span_attribute(span, f"{prefix}.content", message.get("content"))
                function_call = message.get("function_call")
                if function_call:
                    if is_openai_v1() and not isinstance(function_call, dict):
                        function_call = function_call.__dict__

                    _set_span_attribute(
                        span, f"{prefix}.function_call.name", function_call.get("name")
                    )
                    _set_span_attribute(
                        span,
                        f"{prefix}.function_call.arguments",
                        function_call.get("arguments"),
                    )
        elif llm_request_type == LLMRequestTypeValues.COMPLETION:
            _set_span_attribute(span, f"{prefix}.content", choice.get("text"))


def _set_response_attributes(span, llm_request_type, response):
    _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, response.get("model"))
    if should_send_prompts():
        _set_span_completions(span, llm_request_type, response.get("choices"))

    usage = response.get("usage")
    if usage is not None:
        if is_openai_v1() and not isinstance(usage, dict):
            usage = usage.__dict__

        _set_span_attribute(
            span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.get("total_tokens")
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            usage.get("completion_tokens"),
        )
        _set_span_attribute(
            span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.get("prompt_tokens")
        )

    return


def _build_from_streaming_response(span, llm_request_type, response):
    complete_response = {"choices": [], "model": ""}
    for item in response:
        if is_openai_v1():
            item = item.__dict__

        for choice in item.get("choices"):
            if is_openai_v1():
                choice = choice.__dict__

            index = choice.get("index")
            if len(complete_response.get("choices")) <= index:
                complete_response["choices"].append(
                    {"index": index, "message": {"content": "", "role": ""}}
                    if llm_request_type == LLMRequestTypeValues.CHAT
                    else {"index": index, "text": ""}
                )
            complete_choice = complete_response.get("choices")[index]
            if choice.get("finish_reason"):
                complete_choice["finish_reason"] = choice.get("finish_reason")
            if llm_request_type == LLMRequestTypeValues.CHAT:
                delta = choice.get("delta")
                if is_openai_v1():
                    delta = delta.__dict__

                if delta.get("content"):
                    complete_choice["message"]["content"] += delta.get("content")
                if delta.get("role"):
                    complete_choice["message"]["role"] = delta.get("role")
            else:
                complete_choice["text"] += choice.get("text")

        yield item

    _set_response_attributes(
        span,
        llm_request_type,
        complete_response,
    )
    span.set_status(Status(StatusCode.OK))
    span.end()


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def _llm_request_type_by_module_object(module_name, object_name):
    if is_openai_v1():
        if module_name == "openai.resources.chat.completions":
            return LLMRequestTypeValues.CHAT
        elif module_name == "openai.resources.completions":
            return LLMRequestTypeValues.COMPLETION
        else:
            return LLMRequestTypeValues.UNKNOWN
    else:
        if object_name == "Completion":
            return LLMRequestTypeValues.COMPLETION
        elif object_name == "ChatCompletion":
            return LLMRequestTypeValues.CHAT
        else:
            return LLMRequestTypeValues.UNKNOWN


def is_streaming_response(response):
    return isinstance(response, types.GeneratorType) or (
        is_openai_v1() and isinstance(response, openai.Stream)
    )


@_with_tracer_wrapper
def _wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    llm_request_type = _llm_request_type_by_module_object(
        to_wrap.get("module"), to_wrap.get("object")
    )

    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_VENDOR: "OpenAI",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    )

    if span.is_recording():
        _set_api_attributes(span)
    try:
        if span.is_recording():
            _set_input_attributes(span, llm_request_type, kwargs)

    except Exception as ex:  # pylint: disable=broad-except
        logger.warning(
            "Failed to set input attributes for openai span, error: %s", str(ex)
        )

    response = wrapped(*args, **kwargs)

    if response:
        try:
            if span.is_recording():
                if is_streaming_response(response):
                    return _build_from_streaming_response(
                        span, llm_request_type, response
                    )
                else:
                    _set_response_attributes(
                        span,
                        llm_request_type,
                        response.__dict__ if is_openai_v1() else response,
                    )

        except Exception as ex:  # pylint: disable=broad-except
            logger.warning(
                "Failed to set response attributes for openai span, error: %s",
                str(ex),
            )
        if span.is_recording():
            span.set_status(Status(StatusCode.OK))

    span.end()
    return response


class OpenAISpanAttributes:
    OPENAI_API_VERSION = "openai.api_version"
    OPENAI_API_BASE = "openai.api_base"
    OPENAI_API_TYPE = "openai.api_type"


class OpenAIInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI's client library."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        wrapped_methods = (
            WRAPPED_METHODS_VERSION_1 if is_openai_v1() else WRAPPED_METHODS_VERSION_0
        )
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
        wrapped_methods = (
            WRAPPED_METHODS_VERSION_1 if is_openai_v1() else WRAPPED_METHODS_VERSION_0
        )
        for wrapped_method in wrapped_methods:
            wrap_object = wrapped_method.get("object")
            unwrap(f"openai.{wrap_object}", wrapped_method.get("method"))
