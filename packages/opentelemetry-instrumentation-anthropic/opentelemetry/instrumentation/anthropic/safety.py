from __future__ import annotations

from opentelemetry.instrumentation.fortifyroot import (
    SafetyDecision,
    SafetyLocation,
    clone_value,
    get_object_value,
    run_completion_safety,
    run_prompt_safety,
    set_object_value,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues

PROVIDER = "Anthropic"


def _apply_prompt_safety(span, kwargs, span_name: str):
    try:
        request_type = _request_type(span_name)
        mutated_kwargs = kwargs

        prompt = kwargs.get("prompt")
        if isinstance(prompt, str):
            updated_prompt, changed = _mask_prompt_text(
                span,
                prompt,
                span_name=span_name,
                request_type=request_type,
                segment_index=0,
                segment_role="user",
            )
            if changed:
                mutated_kwargs = dict(kwargs)
                mutated_kwargs["prompt"] = updated_prompt

        system = kwargs.get("system")
        updated_system, system_changed = _mask_prompt_content(
            span,
            system,
            span_name=span_name,
            request_type=request_type,
            segment_index=0,
            segment_role="system",
        )
        if system_changed:
            if mutated_kwargs is kwargs:
                mutated_kwargs = dict(kwargs)
            mutated_kwargs["system"] = updated_system

        messages = kwargs.get("messages")
        if not isinstance(messages, list):
            return mutated_kwargs

        mutated_messages = None
        for index, message in enumerate(messages):
            role = get_object_value(message, "role")
            content = get_object_value(message, "content")
            updated_content, changed = _mask_prompt_content(
                span,
                content,
                span_name=span_name,
                request_type=request_type,
                segment_index=index,
                segment_role=role,
            )
            if not changed:
                continue
            if mutated_messages is None:
                if mutated_kwargs is kwargs:
                    mutated_kwargs = dict(kwargs)
                mutated_messages = clone_value(messages)
                mutated_kwargs["messages"] = mutated_messages
            set_object_value(mutated_messages[index], "content", updated_content)

        return mutated_kwargs
    except Exception:
        return kwargs


def _mask_prompt_content(
    span,
    content,
    *,
    span_name,
    request_type,
    segment_index,
    segment_role,
):
    if isinstance(content, str):
        return _mask_prompt_text(
            span,
            content,
            span_name=span_name,
            request_type=request_type,
            segment_index=segment_index,
            segment_role=segment_role,
        )

    if not isinstance(content, list):
        return content, False

    updated_content = content
    for block_index, block in enumerate(content):
        if get_object_value(block, "type") != "text":
            continue
        text = get_object_value(block, "text")
        if not isinstance(text, str):
            continue
        updated_text, changed = _mask_prompt_text(
            span,
            text,
            span_name=span_name,
            request_type=request_type,
            segment_index=segment_index,
            segment_role=segment_role,
            metadata={"block_index": block_index},
        )
        if not changed:
            continue
        if updated_content is content:
            updated_content = clone_value(content)
        set_object_value(updated_content[block_index], "text", updated_text)

    return updated_content, updated_content is not content


def _apply_completion_safety(span, response, span_name: str):
    try:
        request_type = _request_type(span_name)

        completion = get_object_value(response, "completion")
        if isinstance(completion, str):
            updated_completion, changed = _mask_completion_text(
                span,
                completion,
                span_name=span_name,
                request_type=request_type,
                segment_index=0,
                segment_role="assistant",
            )
            if changed:
                set_object_value(response, "completion", updated_completion)

        content = get_object_value(response, "content")
        if not isinstance(content, list):
            return

        for index, block in enumerate(content):
            block_type = get_object_value(block, "type")
            text_key = None
            role = "assistant"
            if block_type == "text":
                text_key = "text"
            elif block_type == "thinking":
                text_key = "thinking"
                role = "thinking"
            if text_key is None:
                continue
            text = get_object_value(block, text_key)
            if not isinstance(text, str):
                continue
            updated_text, changed = _mask_completion_text(
                span,
                text,
                span_name=span_name,
                request_type=request_type,
                segment_index=index,
                segment_role=role,
            )
            if changed:
                set_object_value(block, text_key, updated_text)
    except Exception:
        return


def _mask_prompt_text(
    span,
    text,
    *,
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
    span,
    text,
    *,
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


def _request_type(span_name: str) -> str:
    if span_name.endswith("completion"):
        return LLMRequestTypeValues.COMPLETION.value
    return LLMRequestTypeValues.CHAT.value


def _resolve_masked_text(original_text, result):
    if result is None or result.overall_action != SafetyDecision.MASK.value:
        return original_text, False
    if result.text == original_text:
        return original_text, False
    return result.text, True
