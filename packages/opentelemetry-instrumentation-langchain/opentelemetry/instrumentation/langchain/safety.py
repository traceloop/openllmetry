from __future__ import annotations

import logging
from typing import Any

from opentelemetry.instrumentation.fortifyroot import (
    SafetyDecision,
    SafetyLocation,
    clone_value,
    get_object_value,
    run_completion_safety,
    run_prompt_safety,
    set_object_value,
)
from opentelemetry.instrumentation.langchain.vendor_detection import (
    detect_vendor_from_class,
)
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv_ai import LLMRequestTypeValues
from wrapt import wrap_function_wrapper

PROVIDER = "Langchain"
logger = logging.getLogger(__name__)


def base_chat_model_generate_wrapper(wrapped, instance, args, kwargs):
    updated_args, updated_kwargs = _apply_chat_prompt_safety(instance, args, kwargs)
    return wrapped(*updated_args, **updated_kwargs)


async def base_chat_model_agenerate_wrapper(wrapped, instance, args, kwargs):
    updated_args, updated_kwargs = _apply_chat_prompt_safety(instance, args, kwargs)
    return await wrapped(*updated_args, **updated_kwargs)


def base_llm_generate_wrapper(wrapped, instance, args, kwargs):
    updated_args, updated_kwargs = _apply_llm_prompt_safety(instance, args, kwargs)
    return wrapped(*updated_args, **updated_kwargs)


async def base_llm_agenerate_wrapper(wrapped, instance, args, kwargs):
    updated_args, updated_kwargs = _apply_llm_prompt_safety(instance, args, kwargs)
    return await wrapped(*updated_args, **updated_kwargs)


def base_chat_model_generate_with_cache_wrapper(wrapped, instance, args, kwargs):
    response = wrapped(*args, **kwargs)
    _apply_chat_result_completion_safety(instance, response)
    return response


async def base_chat_model_agenerate_with_cache_wrapper(
    wrapped, instance, args, kwargs
):
    response = await wrapped(*args, **kwargs)
    _apply_chat_result_completion_safety(instance, response)
    return response


def base_llm_generate_helper_wrapper(wrapped, instance, args, kwargs):
    response = wrapped(*args, **kwargs)
    _apply_llm_result_completion_safety(instance, response)
    return response


async def base_llm_agenerate_helper_wrapper(wrapped, instance, args, kwargs):
    response = await wrapped(*args, **kwargs)
    _apply_llm_result_completion_safety(instance, response)
    return response


def _apply_chat_prompt_safety(instance, args, kwargs):
    messages = args[0] if args else kwargs.get("messages")
    if not isinstance(messages, list):
        return args, kwargs

    provider = _provider_name(instance)
    span_name = f"{instance.__class__.__name__}.chat"
    updated_batches = messages
    changed = False

    for batch_index, batch in enumerate(messages):
        if not isinstance(batch, list):
            continue
        updated_batch = batch
        for message_index, message in enumerate(batch):
            content = get_object_value(message, "content")
            updated_content, content_changed = _mask_prompt_content(
                content,
                provider=provider,
                span_name=span_name,
                request_type=LLMRequestTypeValues.CHAT.value,
                segment_index=message_index,
                segment_role=_message_role(message),
                metadata={"batch_index": batch_index},
            )
            if not content_changed:
                continue
            if updated_batches is messages:
                updated_batches = clone_value(messages)
            if updated_batch is batch:
                updated_batch = updated_batches[batch_index]
            set_object_value(updated_batch[message_index], "content", updated_content)
            changed = True

    if not changed:
        return args, kwargs

    if args:
        updated_args = list(args)
        updated_args[0] = updated_batches
        return tuple(updated_args), kwargs

    updated_kwargs = dict(kwargs)
    updated_kwargs["messages"] = updated_batches
    return args, updated_kwargs


def _apply_llm_prompt_safety(instance, args, kwargs):
    prompts = args[0] if args else kwargs.get("prompts")
    if not isinstance(prompts, list):
        return args, kwargs

    provider = _provider_name(instance)
    span_name = f"{instance.__class__.__name__}.completion"
    updated_prompts = prompts
    changed = False

    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, str):
            continue
        updated_prompt, prompt_changed = _mask_prompt_text(
            prompt,
            provider=provider,
            span_name=span_name,
            request_type=LLMRequestTypeValues.COMPLETION.value,
            segment_index=index,
            segment_role="user",
        )
        if not prompt_changed:
            continue
        if updated_prompts is prompts:
            updated_prompts = list(prompts)
        updated_prompts[index] = updated_prompt
        changed = True

    if not changed:
        return args, kwargs

    if args:
        updated_args = list(args)
        updated_args[0] = updated_prompts
        return tuple(updated_args), kwargs

    updated_kwargs = dict(kwargs)
    updated_kwargs["prompts"] = updated_prompts
    return args, updated_kwargs


def _apply_chat_result_completion_safety(instance, response):
    generations = get_object_value(response, "generations")
    if not isinstance(generations, list):
        return

    provider = _provider_name(instance)
    span_name = f"{instance.__class__.__name__}.chat"
    for index, generation in enumerate(generations):
        message = get_object_value(generation, "message")
        message_content = get_object_value(message, "content") if message is not None else None
        if message is not None:
            updated_content, changed = _mask_completion_content(
                message_content,
                provider=provider,
                span_name=span_name,
                request_type=LLMRequestTypeValues.CHAT.value,
                segment_index=index,
                segment_role="assistant",
            )
            if changed:
                set_object_value(message, "content", updated_content)
                if isinstance(updated_content, str):
                    set_object_value(generation, "text", updated_content)
                    continue
        text = get_object_value(generation, "text")
        if not isinstance(text, str):
            continue
        updated_text, changed = _mask_completion_text(
            text,
            provider=provider,
            span_name=span_name,
            request_type=LLMRequestTypeValues.CHAT.value,
            segment_index=index,
            segment_role="assistant",
        )
        if changed:
            set_object_value(generation, "text", updated_text)
            if message is not None and isinstance(message_content, str):
                set_object_value(message, "content", updated_text)


def _apply_llm_result_completion_safety(instance, response):
    generations = get_object_value(response, "generations")
    if not isinstance(generations, list):
        return

    provider = _provider_name(instance)
    span_name = f"{instance.__class__.__name__}.completion"
    for batch in generations:
        if not isinstance(batch, list):
            continue
        for index, generation in enumerate(batch):
            text = get_object_value(generation, "text")
            if not isinstance(text, str):
                continue
            updated_text, changed = _mask_completion_text(
                text,
                provider=provider,
                span_name=span_name,
                request_type=LLMRequestTypeValues.COMPLETION.value,
                segment_index=index,
                segment_role="assistant",
            )
            if changed:
                set_object_value(generation, "text", updated_text)
                message = get_object_value(generation, "message")
                if message is not None and isinstance(get_object_value(message, "content"), str):
                    set_object_value(message, "content", updated_text)


def _mask_prompt_content(
    content,
    *,
    provider,
    span_name,
    request_type,
    segment_index,
    segment_role,
    metadata=None,
):
    if isinstance(content, str):
        return _mask_prompt_text(
            content,
            provider=provider,
            span_name=span_name,
            request_type=request_type,
            segment_index=segment_index,
            segment_role=segment_role,
            metadata=metadata,
        )

    if not isinstance(content, list):
        return content, False

    updated_content = content
    for block_index, block in enumerate(content):
        if isinstance(block, str):
            updated_text, changed = _mask_prompt_text(
                block,
                provider=provider,
                span_name=span_name,
                request_type=request_type,
                segment_index=segment_index,
                segment_role=segment_role,
                metadata={"block_index": block_index, **(metadata or {})},
            )
            if not changed:
                continue
            if updated_content is content:
                updated_content = clone_value(content)
            updated_content[block_index] = updated_text
            continue

        text = _content_text(block)
        if not isinstance(text, str):
            continue
        updated_text, changed = _mask_prompt_text(
            text,
            provider=provider,
            span_name=span_name,
            request_type=request_type,
            segment_index=segment_index,
            segment_role=segment_role,
            metadata={"block_index": block_index, **(metadata or {})},
        )
        if not changed:
            continue
        if updated_content is content:
            updated_content = clone_value(content)
        _set_content_text(updated_content[block_index], updated_text)

    return updated_content, updated_content is not content


def _mask_completion_content(
    content,
    *,
    provider,
    span_name,
    request_type,
    segment_index,
    segment_role,
):
    if isinstance(content, str):
        return _mask_completion_text(
            content,
            provider=provider,
            span_name=span_name,
            request_type=request_type,
            segment_index=segment_index,
            segment_role=segment_role,
        )

    if not isinstance(content, list):
        return content, False

    updated_content = content
    for block_index, block in enumerate(content):
        if isinstance(block, str):
            updated_text, changed = _mask_completion_text(
                block,
                provider=provider,
                span_name=span_name,
                request_type=request_type,
                segment_index=segment_index,
                segment_role=segment_role,
            )
            if not changed:
                continue
            if updated_content is content:
                updated_content = clone_value(content)
            updated_content[block_index] = updated_text
            continue

        text = _content_text(block)
        if not isinstance(text, str):
            continue
        updated_text, changed = _mask_completion_text(
            text,
            provider=provider,
            span_name=span_name,
            request_type=request_type,
            segment_index=segment_index,
            segment_role=segment_role,
        )
        if not changed:
            continue
        if updated_content is content:
            updated_content = clone_value(content)
        _set_content_text(updated_content[block_index], updated_text)

    return updated_content, updated_content is not content


def _mask_prompt_text(
    text,
    *,
    provider,
    span_name,
    request_type,
    segment_index,
    segment_role,
    metadata=None,
):
    result = run_prompt_safety(
        span=None,
        provider=provider,
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
    provider,
    span_name,
    request_type,
    segment_index,
    segment_role,
):
    result = run_completion_safety(
        span=None,
        provider=provider,
        span_name=span_name,
        text=text,
        location=SafetyLocation.COMPLETION,
        request_type=request_type,
        segment_index=segment_index,
        segment_role=segment_role,
    )
    return _resolve_masked_text(text, result)


def _provider_name(instance) -> str:
    return detect_vendor_from_class(instance.__class__.__name__) or PROVIDER


def _message_role(message) -> str:
    message_type = get_object_value(message, "type")
    if message_type == "human":
        return "user"
    if message_type == "system":
        return "system"
    if message_type == "ai":
        return "assistant"
    if message_type == "tool":
        return "tool"
    return str(get_object_value(message, "role", "unknown")).lower()


def _content_text(block):
    if get_object_value(block, "type") == "text":
        return get_object_value(block, "text")
    if get_object_value(block, "text") is not None:
        return get_object_value(block, "text")
    return None


def _set_content_text(block, value):
    if get_object_value(block, "type") == "text":
        return set_object_value(block, "text", value)
    return set_object_value(block, "text", value)


def _resolve_masked_text(original_text, result):
    if result is None or result.overall_action != SafetyDecision.MASK.value:
        return original_text, False
    if result.text == original_text:
        return original_text, False
    return result.text, True


_WRAPPED_METHODS = (
    (
        "langchain_core.language_models.chat_models",
        "BaseChatModel.generate",
        base_chat_model_generate_wrapper,
    ),
    (
        "langchain_core.language_models.chat_models",
        "BaseChatModel.agenerate",
        base_chat_model_agenerate_wrapper,
    ),
    (
        "langchain_core.language_models.chat_models",
        "BaseChatModel._generate_with_cache",
        base_chat_model_generate_with_cache_wrapper,
    ),
    (
        "langchain_core.language_models.chat_models",
        "BaseChatModel._agenerate_with_cache",
        base_chat_model_agenerate_with_cache_wrapper,
    ),
    (
        "langchain_core.language_models.llms",
        "BaseLLM.generate",
        base_llm_generate_wrapper,
    ),
    (
        "langchain_core.language_models.llms",
        "BaseLLM.agenerate",
        base_llm_agenerate_wrapper,
    ),
    (
        "langchain_core.language_models.llms",
        "BaseLLM._generate_helper",
        base_llm_generate_helper_wrapper,
    ),
    (
        "langchain_core.language_models.llms",
        "BaseLLM._agenerate_helper",
        base_llm_agenerate_helper_wrapper,
    ),
)


def instrument_safety_wrappers():
    for module_name, function_name, wrapper in _WRAPPED_METHODS:
        try:
            wrap_function_wrapper(module_name, function_name, wrapper)
        except Exception:
            logger.warning(
                "Failed to install LangChain safety wrapper for %s.%s",
                module_name,
                function_name,
                exc_info=True,
            )


def uninstrument_safety_wrappers():
    for module_name, function_name, _ in _WRAPPED_METHODS:
        try:
            unwrap(module_name, function_name)
        except Exception:
            logger.debug(
                "Failed to remove LangChain safety wrapper for %s.%s",
                module_name,
                function_name,
                exc_info=True,
            )
