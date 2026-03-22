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

PROVIDER = "MistralAI"


def _apply_prompt_safety(span, kwargs, llm_request_type, span_name):
    try:
        if llm_request_type != LLMRequestTypeValues.CHAT:
            return kwargs

        messages = kwargs.get("messages")
        if not isinstance(messages, list):
            return kwargs

        mutated_kwargs = kwargs
        mutated_messages = None
        for index, message in enumerate(messages):
            updated_content, changed = _mask_prompt_content(
                span,
                get_object_value(message, "content"),
                span_name=span_name,
                segment_index=index,
                segment_role=get_object_value(message, "role"),
                request_type=llm_request_type.value if hasattr(llm_request_type, 'value') else llm_request_type,
            )
            if not changed:
                continue
            if mutated_messages is None:
                mutated_kwargs = dict(kwargs)
                mutated_messages = clone_value(messages)
                mutated_kwargs["messages"] = mutated_messages
            set_object_value(mutated_messages[index], "content", updated_content)

        return mutated_kwargs
    except Exception:
        return kwargs


def _apply_completion_safety(span, response, llm_request_type, span_name):
    try:
        if llm_request_type != LLMRequestTypeValues.CHAT:
            return

        choices = get_object_value(response, "choices") or []
        for index, choice in enumerate(choices):
            message = get_object_value(choice, "message")
            if message is None:
                continue
            updated_content, changed = _mask_completion_content(
                span,
                get_object_value(message, "content"),
                span_name=span_name,
                segment_index=index,
                request_type=llm_request_type.value if hasattr(llm_request_type, 'value') else llm_request_type,
            )
            if changed:
                set_object_value(message, "content", updated_content)
    except Exception:
        return


def _mask_prompt_content(span, content, *, span_name, segment_index, segment_role, request_type=LLMRequestTypeValues.CHAT.value):
    if isinstance(content, str):
        return _mask_prompt_text(
            span,
            content,
            span_name=span_name,
            segment_index=segment_index,
            segment_role=segment_role,
            request_type=request_type,
        )

    if not isinstance(content, list):
        return content, False

    updated_content = content
    for block_index, block in enumerate(content):
        block_type = get_object_value(block, "type")
        block_text = get_object_value(block, "text")
        if block_type not in (None, "text") or not isinstance(block_text, str):
            continue
        updated_text, changed = _mask_prompt_text(
            span,
            block_text,
            span_name=span_name,
            segment_index=segment_index,
            segment_role=segment_role,
            request_type=request_type,
            metadata={"block_index": block_index},
        )
        if not changed:
            continue
        if updated_content is content:
            updated_content = clone_value(content)
        set_object_value(updated_content[block_index], "text", updated_text)

    return updated_content, updated_content is not content


def _mask_completion_content(span, content, *, span_name, segment_index, request_type=LLMRequestTypeValues.CHAT.value):
    if isinstance(content, str):
        return _mask_completion_text(
            span,
            content,
            span_name=span_name,
            segment_index=segment_index,
            request_type=request_type,
        )

    if not isinstance(content, list):
        return content, False

    updated_content = content
    for block_index, block in enumerate(content):
        block_text = get_object_value(block, "text")
        if not isinstance(block_text, str):
            continue
        updated_text, changed = _mask_completion_text(
            span,
            block_text,
            span_name=span_name,
            segment_index=segment_index,
            request_type=request_type,
            metadata={"block_index": block_index},
        )
        if not changed:
            continue
        if updated_content is content:
            updated_content = clone_value(content)
        set_object_value(updated_content[block_index], "text", updated_text)

    return updated_content, updated_content is not content


def _mask_prompt_text(
    span,
    text,
    *,
    span_name,
    segment_index,
    segment_role,
    request_type=LLMRequestTypeValues.CHAT.value,
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


def _mask_completion_text(span, text, *, span_name, segment_index, request_type=LLMRequestTypeValues.CHAT.value, metadata=None):
    result = run_completion_safety(
        span=span,
        provider=PROVIDER,
        span_name=span_name,
        text=text,
        location=SafetyLocation.COMPLETION,
        request_type=request_type,
        segment_index=segment_index,
        segment_role="assistant",
        metadata=metadata,
    )
    return _resolve_masked_text(text, result)


def _resolve_masked_text(original_text, result):
    if result is None or result.overall_action != SafetyDecision.MASK.value:
        return original_text, False
    if result.text == original_text:
        return original_text, False
    return result.text, True
