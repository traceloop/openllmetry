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

PROVIDER = "Ollama"


def _apply_prompt_safety(span, kwargs, llm_request_type, span_name):
    try:
        json_data = kwargs.get("json")
        if not isinstance(json_data, dict):
            return kwargs

        mutated_kwargs = kwargs
        mutated_json = json_data

        prompt = json_data.get("prompt")
        if isinstance(prompt, str):
            updated_prompt, changed = _mask_prompt_text(
                span,
                prompt,
                request_type=llm_request_type.value,
                span_name=span_name,
                segment_index=0,
                segment_role="user",
            )
            if changed:
                mutated_kwargs = dict(kwargs)
                mutated_json = dict(json_data)
                mutated_kwargs["json"] = mutated_json
                mutated_json["prompt"] = updated_prompt

        messages = json_data.get("messages")
        if not isinstance(messages, list):
            return mutated_kwargs

        mutated_messages = None
        for index, message in enumerate(messages):
            updated_content, changed = _mask_prompt_content(
                span,
                get_object_value(message, "content"),
                request_type=llm_request_type.value,
                span_name=span_name,
                segment_index=index,
                segment_role=get_object_value(message, "role"),
            )
            if not changed:
                continue
            if mutated_messages is None:
                if mutated_kwargs is kwargs:
                    mutated_kwargs = dict(kwargs)
                if mutated_json is json_data:
                    mutated_json = dict(json_data)
                    mutated_kwargs["json"] = mutated_json
                mutated_messages = clone_value(messages)
                mutated_json["messages"] = mutated_messages
            set_object_value(mutated_messages[index], "content", updated_content)

        return mutated_kwargs
    except Exception:
        return kwargs


def _apply_completion_safety(span, response, llm_request_type, span_name):
    try:
        if llm_request_type.value == "chat":
            message = get_object_value(response, "message")
            if message is None:
                return
            updated_content, changed = _mask_completion_content(
                span,
                get_object_value(message, "content"),
                request_type=llm_request_type.value,
                span_name=span_name,
                segment_index=0,
            )
            if changed:
                set_object_value(message, "content", updated_content)
            return

        text = get_object_value(response, "response")
        if not isinstance(text, str):
            return
        updated_text, changed = _mask_completion_text(
            span,
            text,
            request_type=llm_request_type.value,
            span_name=span_name,
            segment_index=0,
        )
        if changed:
            set_object_value(response, "response", updated_text)
    except Exception:
        return


def _mask_prompt_content(
    span,
    content,
    *,
    request_type,
    span_name,
    segment_index,
    segment_role,
):
    if isinstance(content, str):
        return _mask_prompt_text(
            span,
            content,
            request_type=request_type,
            span_name=span_name,
            segment_index=segment_index,
            segment_role=segment_role,
        )

    if not isinstance(content, list):
        return content, False

    updated_content = content
    for block_index, block in enumerate(content):
        block_text = get_object_value(block, "text")
        if not isinstance(block_text, str):
            continue
        updated_text, changed = _mask_prompt_text(
            span,
            block_text,
            request_type=request_type,
            span_name=span_name,
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


def _mask_completion_content(span, content, *, request_type, span_name, segment_index):
    if isinstance(content, str):
        return _mask_completion_text(
            span,
            content,
            request_type=request_type,
            span_name=span_name,
            segment_index=segment_index,
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
            request_type=request_type,
            span_name=span_name,
            segment_index=segment_index,
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
    request_type,
    span_name,
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


def _mask_completion_text(span, text, *, request_type, span_name, segment_index, metadata=None):
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
