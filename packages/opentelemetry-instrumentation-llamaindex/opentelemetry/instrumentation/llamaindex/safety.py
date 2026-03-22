from __future__ import annotations

import importlib
import logging
import pkgutil
import threading
from typing import Any

import llama_index.core.llms
from llama_index.core.base.llms.base import BaseLLM
from opentelemetry.instrumentation.fortifyroot import (
    SafetyDecision,
    SafetyLocation,
    clone_value,
    get_prompt_safety_handler,
    get_object_value,
    run_completion_safety,
    run_prompt_safety,
    set_object_value,
)
from opentelemetry.instrumentation.llamaindex.utils import should_send_prompts
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry import trace
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from opentelemetry.instrumentation.utils import unwrap
from wrapt import wrap_function_wrapper

try:
    import llama_index.llms
except Exception:  # pragma: no cover
    llama_index = None

PROVIDER = "LlamaIndex"
_WRAPPERS_INSTALLED = False
_WRAPPERS_LOCK = threading.Lock()
_METHODS = ("chat", "achat", "complete", "acomplete")
_WRAPPED_TARGETS: set[tuple[str, str]] = set()
logger = logging.getLogger(__name__)


def instrument_llm_safety_wrappers():
    global _WRAPPERS_INSTALLED
    with _WRAPPERS_LOCK:
        if _WRAPPERS_INSTALLED:
            return

        _wrap_base_methods()

        for package_name in ("llama_index.core.llms", "llama_index.llms"):
            _wrap_package_classes(package_name)

        _WRAPPERS_INSTALLED = True


def uninstrument_llm_safety_wrappers():
    global _WRAPPERS_INSTALLED
    with _WRAPPERS_LOCK:
        for module_name, target in list(_WRAPPED_TARGETS):
            try:
                unwrap_object = module_name
                unwrap_target = target
                if "." in target:
                    owner_name, unwrap_target = target.rsplit(".", 1)
                    unwrap_object = f"{module_name}.{owner_name}"
                unwrap(unwrap_object, unwrap_target)
            except Exception:
                logger.debug(
                    "failed to unwrap llamaindex safety target %s.%s",
                    module_name,
                    target,
                    exc_info=True,
                )
        _WRAPPED_TARGETS.clear()
        _WRAPPERS_INSTALLED = False


def llm_chat_wrapper(wrapped, instance, args, kwargs):
    updated_args, updated_kwargs = _apply_chat_prompt_safety(instance, args, kwargs)
    return wrapped(*updated_args, **updated_kwargs)


async def llm_achat_wrapper(wrapped, instance, args, kwargs):
    updated_args, updated_kwargs = _apply_chat_prompt_safety(instance, args, kwargs)
    return await wrapped(*updated_args, **updated_kwargs)


def llm_complete_wrapper(wrapped, instance, args, kwargs):
    updated_args, updated_kwargs = _apply_completion_prompt_safety(instance, args, kwargs)
    return wrapped(*updated_args, **updated_kwargs)


async def llm_acomplete_wrapper(wrapped, instance, args, kwargs):
    updated_args, updated_kwargs = _apply_completion_prompt_safety(instance, args, kwargs)
    return await wrapped(*updated_args, **updated_kwargs)


_METHOD_WRAPPERS = {
    "chat": llm_chat_wrapper,
    "achat": llm_achat_wrapper,
    "complete": llm_complete_wrapper,
    "acomplete": llm_acomplete_wrapper,
}


def apply_chat_end_safety(event, span):
    _apply_chat_response_safety(event, span)


def apply_completion_start_span_attributes(event, span):
    if span is None or not span.is_recording():
        return

    model_dict = event.model_dict or {}
    if "llm" in model_dict:
        model_dict = model_dict.get("llm", {})

    span.set_attribute(
        SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.COMPLETION.value
    )
    span.set_attribute(GenAIAttributes.GEN_AI_REQUEST_MODEL, model_dict.get("model"))
    span.set_attribute(
        GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE,
        model_dict.get("temperature"),
    )

    if should_send_prompts():
        span.set_attribute(f"{GenAIAttributes.GEN_AI_PROMPT}.0.role", "user")
        span.set_attribute(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content", event.prompt)


apply_completion_start_attributes = apply_completion_start_span_attributes


def apply_completion_end_safety(event, span):
    if span is None or not span.is_recording():
        return

    response = event.response
    if response is not None and isinstance(get_object_value(response, "text"), str):
        updated_text, changed = _mask_completion_text(
            get_object_value(response, "text"),
            span=span,
            span_name=getattr(span, "name", "llamaindex.completion"),
            request_type=LLMRequestTypeValues.COMPLETION.value,
            segment_index=0,
            segment_role="assistant",
        )
        if changed:
            set_object_value(response, "text", updated_text)

    _set_completion_response_model_attributes(response, span)

    if should_send_prompts():
        span.set_attribute(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role", "assistant")
        span.set_attribute(
            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content",
            get_object_value(response, "text"),
        )


def apply_predict_end_safety(event, span):
    if not isinstance(event.output, str):
        return

    updated_text, changed = _mask_completion_text(
        event.output,
        span=span,
        span_name=getattr(span, "name", "llamaindex.predict"),
        request_type=LLMRequestTypeValues.COMPLETION.value,
        segment_index=0,
        segment_role="assistant",
    )
    if changed:
        event.output = updated_text


def _wrap_package_classes(package_name: str):
    try:
        package = importlib.import_module(package_name)
    except Exception:
        logger.debug("failed to import llamaindex package %s", package_name, exc_info=True)
        return

    for module_info in pkgutil.iter_modules(package.__path__):
        module_name = f"{package_name}.{module_info.name}"
        try:
            module = importlib.import_module(module_name)
        except Exception:
            logger.debug("failed to import llamaindex module %s", module_name, exc_info=True)
            continue
        for _, cls in module.__dict__.items():
            if not isinstance(cls, type):
                continue
            if not issubclass(cls, BaseLLM):
                continue
            for method_name in _METHODS:
                if method_name not in cls.__dict__:
                    continue
                _wrap_target(
                    cls.__module__,
                    f"{cls.__name__}.{method_name}",
                    _METHOD_WRAPPERS[method_name],
                )


def _wrap_base_methods():
    for method_name in _METHODS:
        _wrap_target(
            "llama_index.core.base.llms.base",
            f"BaseLLM.{method_name}",
            _METHOD_WRAPPERS[method_name],
        )


def _wrap_target(module_name: str, target: str, wrapper) -> None:
    wrap_function_wrapper(module_name, target, wrapper)
    _WRAPPED_TARGETS.add((module_name, target))


def _apply_chat_prompt_safety(instance, args, kwargs):
    messages = args[0] if args else kwargs.get("messages")
    if not isinstance(messages, (list, tuple)):
        return args, kwargs
    if get_prompt_safety_handler() is None:
        return args, kwargs

    updated_messages = list(messages)
    changed = False
    span_name = f"{instance.__class__.__name__}.chat"

    for index, message in enumerate(messages):
        candidate = clone_value(message)
        if _mask_chat_message(
            candidate,
            span=trace.get_current_span(),
            span_name=span_name,
            request_type=LLMRequestTypeValues.CHAT.value,
            segment_index=index,
            location=SafetyLocation.PROMPT,
        ):
            if not changed:
                updated_messages = list(messages)
            updated_messages[index] = candidate
            changed = True

    if not changed:
        return args, kwargs

    if args:
        updated_args = list(args)
        updated_args[0] = updated_messages
        return tuple(updated_args), kwargs

    updated_kwargs = dict(kwargs)
    updated_kwargs["messages"] = updated_messages
    return args, updated_kwargs


def _apply_completion_prompt_safety(instance, args, kwargs):
    prompt = args[0] if args else kwargs.get("prompt")
    if not isinstance(prompt, str):
        return args, kwargs

    updated_prompt, changed = _mask_prompt_text(
        prompt,
        span=trace.get_current_span(),
        span_name=f"{instance.__class__.__name__}.completion",
        request_type=LLMRequestTypeValues.COMPLETION.value,
        segment_index=0,
        segment_role="user",
    )
    if not changed:
        return args, kwargs

    if args:
        updated_args = list(args)
        updated_args[0] = updated_prompt
        return tuple(updated_args), kwargs

    updated_kwargs = dict(kwargs)
    updated_kwargs["prompt"] = updated_prompt
    return args, updated_kwargs


def _apply_chat_response_safety(event, span):
    response = event.response
    if response is None:
        return

    message = get_object_value(response, "message")
    if message is None:
        return

    _mask_chat_message(
        message,
        span=span,
        span_name=getattr(span, "name", "llamaindex.chat"),
        request_type=LLMRequestTypeValues.CHAT.value,
        segment_index=0,
        segment_role="assistant",
        location=SafetyLocation.COMPLETION,
    )


def _mask_chat_message(
    message,
    *,
    span,
    span_name,
    request_type,
    segment_index,
    segment_role=None,
    location=None,
):
    changed = False
    role = segment_role or _message_role(message)
    if location is None:
        location = SafetyLocation.PROMPT
    blocks = get_object_value(message, "blocks")
    if isinstance(blocks, list):
        for block_index, block in enumerate(blocks):
            text = _block_text(block)
            if not isinstance(text, str):
                continue
            updated_text, text_changed = _mask_text(
                text,
                span=span,
                span_name=span_name,
                request_type=request_type,
                segment_index=segment_index,
                segment_role=role,
                location=location,
                metadata={"block_index": block_index},
            )
            if not text_changed:
                continue
            _set_block_text(block, updated_text)
            changed = True
        return changed

    content = get_object_value(message, "content")
    if not isinstance(content, str):
        return False
    updated_text, text_changed = _mask_text(
        content,
        span=span,
        span_name=span_name,
        request_type=request_type,
        segment_index=segment_index,
        segment_role=role,
        location=location,
    )
    if text_changed:
        set_object_value(message, "content", updated_text)
        return True
    return False


def _set_completion_response_model_attributes(response, span):
    if response is None:
        return

    raw = get_object_value(response, "raw")
    if raw is None:
        return

    model = get_object_value(raw, "model")
    if model:
        span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_MODEL, model)

    usage = get_object_value(raw, "usage")
    if usage is not None:
        completion_tokens = get_object_value(usage, "completion_tokens")
        prompt_tokens = get_object_value(usage, "prompt_tokens")
        total_tokens = get_object_value(usage, "total_tokens")
        if completion_tokens is not None:
            span.set_attribute(
                GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                int(completion_tokens),
            )
        if prompt_tokens is not None:
            span.set_attribute(
                GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
                int(prompt_tokens),
            )
        if total_tokens is not None:
            span.set_attribute(
                SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
                int(total_tokens),
            )


def _mask_prompt_text(
    text,
    *,
    span,
    span_name,
    request_type,
    segment_index,
    segment_role,
    metadata=None,
):
    result = run_prompt_safety(
        span=span,
        provider=PROVIDER,
        span_name=span_name,
        text=text,
        location=SafetyLocation.PROMPT,
        request_type=request_type,
        segment_index=segment_index,
        segment_role=segment_role,
        metadata=metadata,
    )
    return _resolve_masked_text(text, result)


def _mask_completion_text(
    text,
    *,
    span,
    span_name,
    request_type,
    segment_index,
    segment_role,
):
    result = run_completion_safety(
        span=span,
        provider=PROVIDER,
        span_name=span_name,
        text=text,
        location=SafetyLocation.COMPLETION,
        request_type=request_type,
        segment_index=segment_index,
        segment_role=segment_role,
    )
    return _resolve_masked_text(text, result)


def _mask_text(
    text,
    *,
    span,
    span_name,
    request_type,
    segment_index,
    segment_role,
    location,
    metadata=None,
):
    if location == SafetyLocation.PROMPT:
        return _mask_prompt_text(
            text,
            span=span,
            span_name=span_name,
            request_type=request_type,
            segment_index=segment_index,
            segment_role=segment_role,
            metadata=metadata,
        )
    return _mask_completion_text(
        text,
        span=span,
        span_name=span_name,
        request_type=request_type,
        segment_index=segment_index,
        segment_role=segment_role,
    )


def _block_text(block):
    block_type = get_object_value(block, "block_type")
    if block_type == "text":
        return get_object_value(block, "text")
    if block_type == "thinking":
        return get_object_value(block, "content")
    return None


def _set_block_text(block, value):
    block_type = get_object_value(block, "block_type")
    if block_type == "text":
        return set_object_value(block, "text", value)
    if block_type == "thinking":
        return set_object_value(block, "content", value)
    return False


def _message_role(message) -> str:
    role = get_object_value(message, "role")
    role_value = getattr(role, "value", role)
    return str(role_value or "user").lower()


def _resolve_masked_text(original_text, result):
    if result is None or result.overall_action != SafetyDecision.MASK.value:
        return original_text, False
    if result.text == original_text:
        return original_text, False
    return result.text, True
