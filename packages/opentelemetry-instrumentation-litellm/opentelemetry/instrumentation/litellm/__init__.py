"""OpenTelemetry LiteLLM instrumentation."""

import inspect
import logging
from typing import Collection

from opentelemetry import context as context_api
from opentelemetry.instrumentation.fortifyroot import get_object_value
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.litellm.safety import (
    apply_completion_safety,
    apply_prompt_safety,
    extract_prompt_texts,
    extract_text_content,
)
from opentelemetry.instrumentation.litellm.streaming_safety import (
    is_async_streaming_response,
    is_sync_streaming_response,
    wrap_async_streaming_response,
    wrap_sync_streaming_response,
)
from opentelemetry.instrumentation.litellm.version import __version__
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import (
    LLMRequestTypeValues,
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import SpanKind, Status, StatusCode, get_tracer, use_span
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("litellm >= 1.71.2, < 2",)

_WRAPPED_METHODS = [
    ("litellm", "completion", False, False),
    ("litellm", "acompletion", True, False),
    ("litellm", "text_completion", False, True),
    ("litellm", "atext_completion", True, True),
    ("litellm.main", "completion", False, False),
    ("litellm.main", "acompletion", True, False),
    ("litellm.main", "text_completion", False, True),
    ("litellm.main", "atext_completion", True, True),
]


class LiteLLMInstrumentor(BaseInstrumentor):
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        for module_name, func_name, is_async, is_text_completion in _WRAPPED_METHODS:
            wrapper = (
                _build_async_wrapper(tracer, is_text_completion)
                if is_async
                else _build_sync_wrapper(tracer, is_text_completion)
            )
            wrap_function_wrapper(module_name, func_name, wrapper)

    def _uninstrument(self, **kwargs):
        for module_name, func_name, _, _ in _WRAPPED_METHODS:
            try:
                unwrap(module_name, func_name)
            except Exception:
                logger.debug("Failed to unwrap %s.%s", module_name, func_name)


def _build_sync_wrapper(tracer, is_text_completion):
    def wrapper(wrapped, instance, args, kwargs):
        return _invoke_completion(
            tracer,
            wrapped,
            args,
            kwargs,
            is_text_completion=is_text_completion,
        )

    return wrapper


def _build_async_wrapper(tracer, is_text_completion):
    async def wrapper(wrapped, instance, args, kwargs):
        return await _invoke_acompletion(
            tracer,
            wrapped,
            args,
            kwargs,
            is_text_completion=is_text_completion,
        )

    return wrapper


def _invoke_completion(tracer, wrapped, args, kwargs, *, is_text_completion=False):
    if _should_skip_instrumentation(kwargs):
        return wrapped(*args, **kwargs)

    span_name = _span_name(kwargs, is_text_completion)
    request_type = _request_type(kwargs, is_text_completion)
    attributes = {
        GenAIAttributes.GEN_AI_SYSTEM: "litellm",
        SpanAttributes.LLM_REQUEST_TYPE: request_type,
        SpanAttributes.LLM_IS_STREAMING: bool(kwargs.get("stream")),
    }

    span = tracer.start_span(
        span_name,
        kind=SpanKind.CLIENT,
        attributes=attributes,
    )
    with use_span(span, end_on_exit=False):
        _set_request_attributes(span, args, kwargs, is_text_completion)

        updated_args, updated_kwargs = apply_prompt_safety(
            span, args, kwargs, request_type, span_name
        )
        _set_prompt_attributes(
            span,
            updated_args,
            updated_kwargs,
            request_type,
            is_text_completion,
        )

        token = context_api.attach(
            context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True)
        )
        try:
            response = wrapped(*updated_args, **updated_kwargs)
        except Exception as exc:
            context_api.detach(token)
            _record_span_error(span, exc)
            span.end()
            raise

        if inspect.isawaitable(response):
            context_api.detach(token)
            return _finalize_awaitable_response(
                span,
                response,
                request_type,
                span_name,
            )

        context_api.detach(token)
        if is_sync_streaming_response(response):
            return wrap_sync_streaming_response(
                span,
                response,
                request_type,
                span_name,
                _set_response_attributes,
            )
        return _finalize_response(span, response, request_type, span_name)


async def _invoke_acompletion(
    tracer,
    wrapped,
    args,
    kwargs,
    *,
    is_text_completion=False,
):
    if _should_skip_instrumentation(kwargs):
        return await wrapped(*args, **kwargs)

    span_name = _span_name(kwargs, is_text_completion)
    request_type = _request_type(kwargs, is_text_completion)
    attributes = {
        GenAIAttributes.GEN_AI_SYSTEM: "litellm",
        SpanAttributes.LLM_REQUEST_TYPE: request_type,
        SpanAttributes.LLM_IS_STREAMING: bool(kwargs.get("stream")),
    }

    span = tracer.start_span(
        span_name,
        kind=SpanKind.CLIENT,
        attributes=attributes,
    )
    with use_span(span, end_on_exit=False):
        _set_request_attributes(span, args, kwargs, is_text_completion)

        updated_args, updated_kwargs = apply_prompt_safety(
            span, args, kwargs, request_type, span_name
        )
        _set_prompt_attributes(
            span,
            updated_args,
            updated_kwargs,
            request_type,
            is_text_completion,
        )

        token = context_api.attach(
            context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True)
        )
        try:
            response = await wrapped(*updated_args, **updated_kwargs)
        except Exception as exc:
            _record_span_error(span, exc)
            span.end()
            raise
        finally:
            context_api.detach(token)

        if is_async_streaming_response(response):
            return wrap_async_streaming_response(
                span,
                response,
                request_type,
                span_name,
                _set_response_attributes,
            )
        return _finalize_response(span, response, request_type, span_name)


def _should_skip_instrumentation(kwargs):
    return bool(
        context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)
        or context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
    )


def _span_name(kwargs, is_text_completion):
    if is_text_completion or kwargs.get("text_completion"):
        return "litellm.text_completion"
    return "litellm.completion"


def _request_type(kwargs, is_text_completion):
    if is_text_completion or kwargs.get("text_completion"):
        return LLMRequestTypeValues.COMPLETION.value
    return LLMRequestTypeValues.CHAT.value


def _set_request_attributes(span, args, kwargs, is_text_completion):
    model = kwargs.get("model")
    if model is None and args:
        if is_text_completion:
            model = args[1] if len(args) > 1 else None
        else:
            model = args[0]
    if model is not None:
        span.set_attribute(GenAIAttributes.GEN_AI_REQUEST_MODEL, str(model))

    user = kwargs.get("user")
    if user is not None:
        span.set_attribute(SpanAttributes.LLM_USER, str(user))

    custom_provider = kwargs.get("custom_llm_provider")
    if custom_provider is not None:
        span.set_attribute("litellm.request.provider", str(custom_provider))


def _set_prompt_attributes(span, args, kwargs, request_type, is_text_completion):
    if request_type == LLMRequestTypeValues.COMPLETION.value:
        prompt = kwargs.get("prompt")
        if prompt is None and args:
            prompt = args[0]

        prompt_texts = extract_prompt_texts(prompt)
        for index, text in enumerate(prompt_texts):
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.role", "user")
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.content", text)
        return

    messages = kwargs.get("messages")
    if messages is None and len(args) > 1:
        messages = args[1]
    if not isinstance(messages, list):
        return

    for index, message in enumerate(messages):
        role = get_object_value(message, "role")
        content = extract_text_content(get_object_value(message, "content"))
        if role is not None:
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.role", str(role))
        if content:
            span.set_attribute(
                f"{SpanAttributes.LLM_PROMPTS}.{index}.content",
                content,
            )


def _set_response_attributes(span, response):
    response_model = get_object_value(response, "model")
    if response_model is not None:
        span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_MODEL, str(response_model))

    usage = get_object_value(response, "usage")
    input_tokens = get_object_value(usage, "prompt_tokens")
    output_tokens = get_object_value(usage, "completion_tokens")
    total_tokens = get_object_value(usage, "total_tokens")

    if input_tokens is not None:
        span.set_attribute(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, int(input_tokens))
    if output_tokens is not None:
        span.set_attribute(
            GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
            int(output_tokens),
        )
    if total_tokens is not None:
        span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, int(total_tokens))

    choices = get_object_value(response, "choices") or []
    for index, choice in enumerate(choices):
        finish_reason = get_object_value(choice, "finish_reason")
        if finish_reason is not None:
            span.set_attribute(
                f"{SpanAttributes.LLM_COMPLETIONS}.{index}.finish_reason",
                str(finish_reason),
            )

        message = get_object_value(choice, "message")
        role = get_object_value(message, "role")
        content = extract_text_content(get_object_value(message, "content"))
        if role is not None:
            span.set_attribute(
                f"{SpanAttributes.LLM_COMPLETIONS}.{index}.role",
                str(role),
            )
        if content:
            span.set_attribute(
                f"{SpanAttributes.LLM_COMPLETIONS}.{index}.content",
                content,
            )
            continue

        text = get_object_value(choice, "text")
        if isinstance(text, str) and text:
            span.set_attribute(
                f"{SpanAttributes.LLM_COMPLETIONS}.{index}.content",
                text,
            )


async def _finalize_awaitable_response(
    span,
    response,
    request_type,
    span_name,
):
    token = context_api.attach(
        context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True)
    )
    try:
        awaited_response = await response
    except Exception as exc:
        _record_span_error(span, exc)
        span.end()
        raise
    finally:
        context_api.detach(token)

    return _finalize_response(span, awaited_response, request_type, span_name)


def _finalize_response(span, response, request_type, span_name):
    try:
        apply_completion_safety(span, response, request_type, span_name)
        _set_response_attributes(span, response)
        span.set_status(Status(StatusCode.OK))
        return response
    finally:
        span.end()


def _record_span_error(span, exc):
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))
__all__ = [
    "LiteLLMInstrumentor",
    "_invoke_acompletion",
    "_invoke_completion",
]
