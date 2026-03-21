from __future__ import annotations

from typing import Any

from opentelemetry.instrumentation.fortifyroot import (
    clone_value,
    get_object_value,
    set_object_value,
)
from opentelemetry.instrumentation.fortifyroot.text_streaming import (
    CompletionTextStreamGroup,
)
from opentelemetry.instrumentation.openai.shared.safety_common import (
    CHAT_PROVIDER,
    CHAT_SPAN_NAME,
    mask_completion_text,
    mask_prompt_text,
    request_type,
)


def apply_assistant_instruction_prompt_safety(span, kwargs: dict[str, Any]) -> dict[str, Any]:
    instructions = kwargs.get("instructions")
    if not isinstance(instructions, str):
        return kwargs

    masked_instructions, changed = mask_prompt_text(
        span,
        instructions,
        span_name=CHAT_SPAN_NAME,
        segment_index=0,
        segment_role="system",
    )
    if not changed:
        return kwargs

    updated_kwargs = dict(kwargs)
    updated_kwargs["instructions"] = masked_instructions
    return updated_kwargs


def apply_assistant_message_prompt_safety(span, args, kwargs):
    content = kwargs.get("content")
    content_in_kwargs = True
    if content is None and args:
        content = args[0]
        content_in_kwargs = False

    if content is None:
        return args, kwargs

    role = str(kwargs.get("role", "user") or "user").lower()
    masked_content, changed = _mask_assistant_content(
        span,
        content,
        location="prompt",
        segment_index=0,
        segment_role=role,
    )
    if not changed:
        return args, kwargs

    if content_in_kwargs:
        updated_kwargs = dict(kwargs)
        updated_kwargs["content"] = masked_content
        return args, updated_kwargs

    updated_args = list(args)
    updated_args[0] = masked_content
    return tuple(updated_args), kwargs


def apply_assistant_messages_list_safety(span, response):
    data = get_object_value(response, "data")
    if not isinstance(data, list):
        return response

    for message_index, message in enumerate(data):
        role = str(get_object_value(message, "role", "user") or "user").lower()
        content = get_object_value(message, "content")
        masked_content, changed = _mask_assistant_content(
            span,
            content,
            location="completion" if role == "assistant" else "prompt",
            segment_index=message_index,
            segment_role=role,
        )
        if changed:
            set_object_value(message, "content", masked_content)
    return response


class AssistantStreamingSafety:
    def __init__(self, span):
        self._span = span
        self._streams = CompletionTextStreamGroup(
            span=span,
            provider=CHAT_PROVIDER,
            span_name=CHAT_SPAN_NAME,
            request_type=request_type(CHAT_SPAN_NAME),
        )
        self._accumulated_text: dict[int, str] = {}

    def process_text_delta(self, delta, snapshot, *, text_index: int) -> str:
        value = get_object_value(delta, "value")
        if not isinstance(value, str):
            return ""

        masked = self._streams.process(
            ("assistant-text", text_index),
            value,
            segment_index=text_index,
            segment_role="assistant",
        )
        set_object_value(delta, "value", masked)
        full_text = self._accumulated_text.get(text_index, "") + masked
        self._accumulated_text[text_index] = full_text
        if snapshot is not None:
            set_object_value(snapshot, "value", full_text)
        return masked

    def flush_text(self, text, *, text_index: int) -> str:
        tail = self._streams.flush(("assistant-text", text_index))
        full_text = self._accumulated_text.get(text_index, "")
        if tail:
            full_text += tail
            self._accumulated_text[text_index] = full_text
        if text is not None:
            set_object_value(text, "value", full_text or get_object_value(text, "value"))
        return tail

    def apply_message_safety(self, message, *, text_index: int) -> None:
        content = get_object_value(message, "content")
        if not isinstance(content, list):
            return
        accumulated = self._accumulated_text.get(text_index)
        for block in content:
            text = get_object_value(block, "text")
            if text is None:
                continue
            if accumulated is not None:
                set_object_value(text, "value", accumulated)
            else:
                value = get_object_value(text, "value")
                if isinstance(value, str):
                    masked_value, changed = mask_completion_text(
                        self._span,
                        value,
                        span_name=CHAT_SPAN_NAME,
                        segment_index=text_index,
                    )
                    if changed:
                        set_object_value(text, "value", masked_value)
            break


def _mask_assistant_content(
    span,
    content,
    *,
    location: str,
    segment_index: int,
    segment_role: str,
):
    if isinstance(content, str):
        return _mask_assistant_text(
            span,
            content,
            location=location,
            segment_index=segment_index,
            segment_role=segment_role,
        )

    if not isinstance(content, list):
        return content, False

    updated_content = content
    changed = False

    for block_index, block in enumerate(content):
        if isinstance(block, str):
            masked_text, text_changed = _mask_assistant_text(
                span,
                block,
                location=location,
                segment_index=segment_index,
                segment_role=segment_role,
            )
            if not text_changed:
                continue
            if updated_content is content:
                updated_content = clone_value(content)
            updated_content[block_index] = masked_text
            changed = True
            continue

        text = _assistant_block_text(block)
        if not isinstance(text, str):
            continue
        masked_text, text_changed = _mask_assistant_text(
            span,
            text,
            location=location,
            segment_index=segment_index,
            segment_role=segment_role,
        )
        if not text_changed:
            continue
        if updated_content is content:
            updated_content = clone_value(content)
        _set_assistant_block_text(updated_content[block_index], masked_text)
        changed = True

    return updated_content, changed


def _mask_assistant_text(
    span,
    text: str,
    *,
    location: str,
    segment_index: int,
    segment_role: str,
):
    if location == "prompt":
        return mask_prompt_text(
            span,
            text,
            span_name=CHAT_SPAN_NAME,
            segment_index=segment_index,
            segment_role=segment_role,
        )
    return mask_completion_text(
        span,
        text,
        span_name=CHAT_SPAN_NAME,
        segment_index=segment_index,
    )


def _assistant_block_text(block):
    text = get_object_value(block, "text")
    if text is None:
        return None
    return get_object_value(text, "value", text)


def _set_assistant_block_text(block, value: str) -> None:
    text = get_object_value(block, "text")
    if text is None:
        set_object_value(block, "text", value)
        return
    if not set_object_value(text, "value", value):
        set_object_value(block, "text", value)
