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

PROVIDER = "VertexAI"


def _apply_prompt_safety(span, args, kwargs, span_name):
    try:
        updated_args = args
        updated_kwargs = kwargs

        if args:
            masked_arg, changed = _mask_prompt_value(
                span,
                args[0],
                span_name=span_name,
                segment_index=0,
                segment_role="user",
            )
            if changed:
                updated_args = (masked_arg, *args[1:])

        if "contents" in kwargs:
            masked_contents, changed = _mask_prompt_value(
                span,
                kwargs.get("contents"),
                span_name=span_name,
                segment_index=0,
                segment_role="user",
            )
            if changed:
                updated_kwargs = dict(kwargs)
                updated_kwargs["contents"] = masked_contents

        return updated_args, updated_kwargs
    except Exception:
        return args, kwargs


def _apply_completion_safety(span, response, span_name):
    try:
        candidates = get_object_value(response, "candidates")
        if not isinstance(candidates, list):
            return

        for candidate_index, candidate in enumerate(candidates):
            content = get_object_value(candidate, "content")
            parts = get_object_value(content, "parts")
            if isinstance(parts, list):
                for part_index, part in enumerate(parts):
                    part_text = get_object_value(part, "text")
                    if not isinstance(part_text, str):
                        continue
                    updated_part_text, changed = _mask_completion_text(
                        span,
                        part_text,
                        span_name=span_name,
                        segment_index=candidate_index,
                        metadata={"part_index": part_index},
                    )
                    if changed:
                        set_object_value(part, "text", updated_part_text)
                continue

            candidate_text = get_object_value(candidate, "text")
            if not isinstance(candidate_text, str):
                continue
            updated_candidate_text, changed = _mask_completion_text(
                span,
                candidate_text,
                span_name=span_name,
                segment_index=candidate_index,
            )
            if changed:
                set_object_value(candidate, "text", updated_candidate_text)
    except Exception:
        return


def _mask_prompt_value(span, value, *, span_name, segment_index, segment_role):
    if isinstance(value, str):
        return _mask_prompt_text(
            span,
            value,
            span_name=span_name,
            segment_index=segment_index,
            segment_role=segment_role,
        )

    if isinstance(value, list):
        updated_value = value
        for index, item in enumerate(value):
            masked_item, changed = _mask_prompt_value(
                span,
                item,
                span_name=span_name,
                segment_index=index,
                segment_role=segment_role,
            )
            if not changed:
                continue
            if updated_value is value:
                updated_value = clone_value(value)
            updated_value[index] = masked_item
        return updated_value, updated_value is not value

    parts = get_object_value(value, "parts")
    if isinstance(parts, list):
        updated_parts = parts
        updated_value = value
        for index, part in enumerate(parts):
            masked_part, changed = _mask_prompt_value(
                span,
                part,
                span_name=span_name,
                segment_index=index,
                segment_role=segment_role,
            )
            if not changed:
                continue
            if updated_parts is parts:
                updated_parts = clone_value(parts)
                updated_value = clone_value(value)
                set_object_value(updated_value, "parts", updated_parts)
            updated_parts[index] = masked_part
        return updated_value, updated_parts is not parts

    text = get_object_value(value, "text")
    if not isinstance(text, str):
        return value, False

    updated_text, changed = _mask_prompt_text(
        span,
        text,
        span_name=span_name,
        segment_index=segment_index,
        segment_role=segment_role,
    )
    if not changed:
        return value, False
    updated_value = clone_value(value)
    set_object_value(updated_value, "text", updated_text)
    return updated_value, True


def _mask_prompt_text(span, text, *, span_name, segment_index, segment_role):
    result = run_prompt_safety(
        span=span,
        provider=PROVIDER,
        span_name=span_name,
        text=text,
        location=SafetyLocation.PROMPT,
        request_type=LLMRequestTypeValues.COMPLETION.value,
        segment_index=segment_index,
        segment_role=segment_role,
    )
    return _resolve_masked_text(text, result)


def _mask_completion_text(span, text, *, span_name, segment_index, metadata=None):
    result = run_completion_safety(
        span=span,
        provider=PROVIDER,
        span_name=span_name,
        text=text,
        location=SafetyLocation.COMPLETION,
        request_type=LLMRequestTypeValues.COMPLETION.value,
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
