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

PROVIDER = "Transformers"
_PROMPT_KWARGS = ("text_inputs", "args")


def _apply_prompt_safety(span, args, kwargs, span_name):
    try:
        prompts, source = _get_prompt_input(args, kwargs)
        if source is None:
            return args, kwargs

        updated_prompts, changed = _mask_prompt_value(
            span,
            prompts,
            span_name=span_name,
            segment_index=0,
            segment_role="user",
        )
        if not changed:
            return args, kwargs

        if source == "positional_args":
            return (updated_prompts, *args[1:]), kwargs

        mutated_kwargs = dict(kwargs)
        mutated_kwargs[source] = updated_prompts
        return args, mutated_kwargs
    except Exception:
        return args, kwargs


def _apply_completion_safety(span, response, span_name):
    try:
        _mask_completion_value(span, response, span_name=span_name, segment_index=0)
    except Exception:
        return


def _get_prompt_input(args, kwargs):
    if args:
        return args[0], "positional_args"
    for key in _PROMPT_KWARGS:
        if key in kwargs:
            return kwargs.get(key), key
    return None, None


def _mask_prompt_value(span, value, *, span_name, segment_index, segment_role):
    if isinstance(value, str):
        return _mask_prompt_text(
            span,
            value,
            span_name=span_name,
            segment_index=segment_index,
            segment_role=segment_role,
        )

    if isinstance(value, dict):
        content = get_object_value(value, "content")
        if not isinstance(content, str):
            return value, False

        updated_content, changed = _mask_prompt_text(
            span,
            content,
            span_name=span_name,
            segment_index=segment_index,
            segment_role=get_object_value(value, "role") or segment_role,
        )
        if not changed:
            return value, False
        updated_value = clone_value(value)
        set_object_value(updated_value, "content", updated_content)
        return updated_value, True

    if not isinstance(value, list):
        return value, False

    updated_value = value
    for index, item in enumerate(value):
        updated_item, changed = _mask_prompt_value(
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
        updated_value[index] = updated_item

    return updated_value, updated_value is not value


def _mask_completion_value(span, value, *, span_name, segment_index):
    if isinstance(value, str):
        return _mask_completion_text(
            span,
            value,
            span_name=span_name,
            segment_index=segment_index,
        )

    if isinstance(value, dict):
        changed_any = False
        generated_text = get_object_value(value, "generated_text")
        if isinstance(generated_text, (str, list, dict)):
            updated_generated_text, changed = _mask_completion_value(
                span,
                generated_text,
                span_name=span_name,
                segment_index=segment_index,
            )
            if changed:
                set_object_value(value, "generated_text", updated_generated_text)
                changed_any = True

        content = get_object_value(value, "content")
        if isinstance(content, str):
            updated_content, changed = _mask_completion_text(
                span,
                content,
                span_name=span_name,
                segment_index=segment_index,
            )
            if changed:
                set_object_value(value, "content", updated_content)
                changed_any = True

        return value, changed_any

    if not isinstance(value, list):
        return value, False

    changed_any = False
    for index, item in enumerate(value):
        updated_item, changed = _mask_completion_value(
            span,
            item,
            span_name=span_name,
            segment_index=index,
        )
        if not changed:
            continue
        value[index] = updated_item
        changed_any = True

    return value, changed_any


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


def _mask_completion_text(span, text, *, span_name, segment_index):
    result = run_completion_safety(
        span=span,
        provider=PROVIDER,
        span_name=span_name,
        text=text,
        location=SafetyLocation.COMPLETION,
        request_type=LLMRequestTypeValues.COMPLETION.value,
        segment_index=segment_index,
        segment_role="assistant",
    )
    return _resolve_masked_text(text, result)


def _resolve_masked_text(original_text, result):
    if result is None or result.overall_action != SafetyDecision.MASK.value:
        return original_text, False
    if result.text == original_text:
        return original_text, False
    return result.text, True
