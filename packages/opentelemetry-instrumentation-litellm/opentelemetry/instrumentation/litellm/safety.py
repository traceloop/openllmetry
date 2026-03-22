from __future__ import annotations

import logging

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

PROVIDER = "LiteLLM"
logger = logging.getLogger(__name__)


def apply_prompt_safety(span, args, kwargs, request_type, span_name):
    try:
        messages, source = _get_messages(args, kwargs)
        if isinstance(messages, list):
            return _apply_messages_prompt_safety(
                span,
                args,
                kwargs,
                messages,
                source,
                request_type,
                span_name,
            )

        if request_type != LLMRequestTypeValues.COMPLETION.value:
            return args, kwargs

        return _apply_text_prompt_safety(span, args, kwargs, request_type, span_name)
    except Exception:
        logger.warning("safety prompt error", exc_info=True)
        return args, kwargs


def _apply_messages_prompt_safety(
    span,
    args,
    kwargs,
    messages,
    source,
    request_type,
    span_name,
):
    updated_messages = messages
    changed = False

    for index, message in enumerate(messages):
        content = get_object_value(message, "content")
        updated_content, content_changed = _mask_prompt_content(
            span,
            content,
            span_name=span_name,
            request_type=request_type,
            segment_index=index,
            segment_role=get_object_value(message, "role") or "user",
        )
        if not content_changed:
            continue
        if updated_messages is messages:
            updated_messages = clone_value(messages)
        set_object_value(updated_messages[index], "content", updated_content)
        changed = True

    if not changed:
        return args, kwargs

    if source == "args":
        updated_args = list(args)
        updated_args[1] = updated_messages
        return tuple(updated_args), kwargs

    updated_kwargs = dict(kwargs)
    updated_kwargs["messages"] = updated_messages
    return args, updated_kwargs


def _apply_text_prompt_safety(span, args, kwargs, request_type, span_name):
    prompt, source = _get_prompt(args, kwargs)
    updated_prompt, changed = _mask_text_prompt_value(
        span,
        prompt,
        span_name=span_name,
        request_type=request_type,
    )
    if not changed:
        return args, kwargs

    if source == "args":
        updated_args = list(args)
        updated_args[0] = updated_prompt
        return tuple(updated_args), kwargs

    updated_kwargs = dict(kwargs)
    updated_kwargs["prompt"] = updated_prompt
    return args, updated_kwargs


def _mask_text_prompt_value(span, value, *, span_name, request_type):
    if isinstance(value, str):
        return _mask_prompt_text(
            span,
            value,
            span_name=span_name,
            request_type=request_type,
            segment_index=0,
            segment_role="user",
        )

    if not isinstance(value, list):
        return value, False

    updated_value = value
    for index, item in enumerate(value):
        updated_item, changed = _mask_text_prompt_item(
            span,
            item,
            span_name=span_name,
            request_type=request_type,
            segment_index=index,
        )
        if not changed:
            continue
        if updated_value is value:
            updated_value = clone_value(value)
        updated_value[index] = updated_item

    return updated_value, updated_value is not value


def _mask_text_prompt_item(
    span,
    value,
    *,
    span_name,
    request_type,
    segment_index,
    metadata=None,
):
    if isinstance(value, str):
        return _mask_prompt_text(
            span,
            value,
            span_name=span_name,
            request_type=request_type,
            segment_index=segment_index,
            segment_role="user",
            metadata=metadata,
        )

    if not isinstance(value, list):
        return value, False

    updated_value = value
    for index, item in enumerate(value):
        updated_item, changed = _mask_text_prompt_item(
            span,
            item,
            span_name=span_name,
            request_type=request_type,
            segment_index=segment_index,
            metadata={"nested_index": index, **(metadata or {})},
        )
        if not changed:
            continue
        if updated_value is value:
            updated_value = clone_value(value)
        updated_value[index] = updated_item

    return updated_value, updated_value is not value


def extract_prompt_texts(prompt):
    texts = []
    _collect_prompt_texts(prompt, texts)
    return texts


def _collect_prompt_texts(value, texts):
    if isinstance(value, str):
        texts.append(value)
        return

    if not isinstance(value, list):
        return

    for item in value:
        _collect_prompt_texts(item, texts)


def _get_prompt(args, kwargs):
    if "prompt" in kwargs:
        return kwargs.get("prompt"), "kwargs"
    if args:
        return args[0], "args"
    return None, None


def apply_completion_safety(span, response, request_type, span_name):
    try:
        choices = get_object_value(response, "choices") or []
        for index, choice in enumerate(choices):
            message = get_object_value(choice, "message")
            message_content = get_object_value(message, "content") if message is not None else None
            if message is not None:
                updated_content, content_changed = _mask_completion_content(
                    span,
                    message_content,
                    span_name=span_name,
                    request_type=request_type,
                    segment_index=index,
                )
                if content_changed:
                    set_object_value(message, "content", updated_content)
                    if isinstance(updated_content, str):
                        set_object_value(choice, "text", updated_content)
                        continue

            text = get_object_value(choice, "text")
            if not isinstance(text, str):
                continue
            updated_text, text_changed = _mask_completion_text(
                span,
                text,
                span_name=span_name,
                request_type=request_type,
                segment_index=index,
            )
            if text_changed:
                set_object_value(choice, "text", updated_text)
                if message is not None and isinstance(message_content, str):
                    set_object_value(message, "content", updated_text)
    except Exception:
        logger.warning("safety completion error", exc_info=True)
        return


def _get_messages(args, kwargs):
    if len(args) > 1:
        return args[1], "args"
    return kwargs.get("messages"), "kwargs"


def _mask_prompt_content(
    span,
    content,
    *,
    span_name,
    request_type,
    segment_index,
    segment_role,
    metadata=None,
):
    if isinstance(content, str):
        return _mask_prompt_text(
            span,
            content,
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
                span,
                block,
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

        block_type = get_object_value(block, "type")
        block_text = get_object_value(block, "text")
        if block_type not in (None, "text", "input_text") or not isinstance(block_text, str):
            continue
        updated_text, changed = _mask_prompt_text(
            span,
            block_text,
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
        set_object_value(updated_content[block_index], "text", updated_text)

    return updated_content, updated_content is not content


def _mask_completion_content(
    span,
    content,
    *,
    span_name,
    request_type,
    segment_index,
    metadata=None,
):
    if isinstance(content, str):
        return _mask_completion_text(
            span,
            content,
            span_name=span_name,
            request_type=request_type,
            segment_index=segment_index,
            metadata=metadata,
        )

    if not isinstance(content, list):
        return content, False

    updated_content = content
    for block_index, block in enumerate(content):
        if isinstance(block, str):
            updated_text, changed = _mask_completion_text(
                span,
                block,
                span_name=span_name,
                request_type=request_type,
                segment_index=segment_index,
                metadata={"block_index": block_index, **(metadata or {})},
            )
            if not changed:
                continue
            if updated_content is content:
                updated_content = clone_value(content)
            updated_content[block_index] = updated_text
            continue

        block_type = get_object_value(block, "type")
        block_text = get_object_value(block, "text")
        if block_type not in (None, "text", "output_text") or not isinstance(block_text, str):
            continue
        updated_text, changed = _mask_completion_text(
            span,
            block_text,
            span_name=span_name,
            request_type=request_type,
            segment_index=segment_index,
            metadata={"block_index": block_index, **(metadata or {})},
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
    metadata=None,
):
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


def extract_text_content(content):
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return None

    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        block_type = get_object_value(block, "type")
        block_text = get_object_value(block, "text")
        if block_type in (None, "text", "input_text", "output_text") and isinstance(
            block_text, str
        ):
            parts.append(block_text)

    if not parts:
        return None
    return "\n".join(parts)


def _resolve_masked_text(original_text, result):
    if result is None or result.overall_action != SafetyDecision.MASK.value:
        return original_text, False
    if result.text == original_text:
        return original_text, False
    return result.text, True
