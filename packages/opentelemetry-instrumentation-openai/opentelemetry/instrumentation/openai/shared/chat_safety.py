from __future__ import annotations

from opentelemetry.instrumentation.fortifyroot import (
    clone_value,
    get_object_value,
    set_object_value,
)
from opentelemetry.instrumentation.openai.shared.safety_common import (
    CHAT_SPAN_NAME,
    mask_completion_text,
    mask_prompt_text,
)


def _apply_prompt_safety(span, kwargs):
    try:
        messages = kwargs.get("messages")
        if not isinstance(messages, list):
            return kwargs

        mutated_kwargs = kwargs
        mutated_messages = None
        for index, message in enumerate(messages):
            role = get_object_value(message, "role")
            content = get_object_value(message, "content")
            updated_content, changed = _mask_prompt_content(
                span,
                content,
                message_index=index,
                message_role=role,
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


def _mask_prompt_content(span, content, *, message_index, message_role):
    if isinstance(content, str):
        return mask_prompt_text(
            span,
            content,
            span_name=CHAT_SPAN_NAME,
            segment_index=message_index,
            segment_role=message_role,
        )

    if not isinstance(content, list):
        return content, False

    updated_content = content
    for block_index, block in enumerate(content):
        block_type = get_object_value(block, "type")
        block_text = get_object_value(block, "text")
        if block_type not in ("text", "input_text") or not isinstance(block_text, str):
            continue
        resolved_text, changed = mask_prompt_text(
            span,
            block_text,
            span_name=CHAT_SPAN_NAME,
            segment_index=message_index,
            segment_role=message_role,
            metadata={"block_index": block_index},
        )
        if not changed:
            continue
        if updated_content is content:
            updated_content = clone_value(content)
        set_object_value(updated_content[block_index], "text", resolved_text)

    return updated_content, updated_content is not content


def _apply_completion_safety(span, response):
    try:
        choices = getattr(response, "choices", None)
        if not choices:
            return

        for choice_index, choice in enumerate(choices):
            message = get_object_value(choice, "message")
            if message is None:
                continue
            content = get_object_value(message, "content")
            updated_content, changed = _mask_completion_content(
                span,
                content,
                choice_index=choice_index,
            )
            if changed:
                set_object_value(message, "content", updated_content)
    except Exception:
        return


def _mask_completion_content(span, content, *, choice_index):
    if isinstance(content, str):
        return mask_completion_text(
            span,
            content,
            span_name=CHAT_SPAN_NAME,
            segment_index=choice_index,
        )

    if not isinstance(content, list):
        return content, False

    updated_content = content
    for block_index, block in enumerate(content):
        block_type = get_object_value(block, "type")
        block_text = get_object_value(block, "text")
        if block_type not in (None, "text", "output_text") or not isinstance(block_text, str):
            continue
        resolved_text, changed = mask_completion_text(
            span,
            block_text,
            span_name=CHAT_SPAN_NAME,
            segment_index=choice_index,
            metadata={"block_index": block_index},
        )
        if not changed:
            continue
        if updated_content is content:
            updated_content = clone_value(content)
        set_object_value(updated_content[block_index], "text", resolved_text)

    return updated_content, updated_content is not content
