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

PROVIDER = "Replicate"
_PROMPT_KEY_MARKERS = ("prompt", "text", "input", "query", "instruction")


def _apply_prompt_safety(span, args, kwargs, span_name):
    try:
        model_input = kwargs.get("input") or (args[1] if len(args) > 1 else None)
        if not isinstance(model_input, dict):
            return args, kwargs

        updated_input, changed = _mask_prompt_value(
            span,
            model_input,
            span_name=span_name,
            segment_index=0,
        )
        if not changed:
            return args, kwargs

        if "input" in kwargs:
            mutated_kwargs = dict(kwargs)
            mutated_kwargs["input"] = updated_input
            return args, mutated_kwargs

        mutated_args = list(args)
        mutated_args[1] = updated_input
        return tuple(mutated_args), kwargs
    except Exception:
        return args, kwargs


def _apply_completion_safety(span, response, span_name):
    try:
        if isinstance(response, list):
            for index, item in enumerate(response):
                if not isinstance(item, str):
                    continue
                updated_item, changed = _mask_completion_text(
                    span,
                    item,
                    span_name=span_name,
                    segment_index=index,
                )
                if changed:
                    response[index] = updated_item
            return response

        if isinstance(response, str):
            updated_response, changed = _mask_completion_text(
                span,
                response,
                span_name=span_name,
                segment_index=0,
            )
            return updated_response if changed else response

        output = get_object_value(response, "output")
        if isinstance(output, list):
            updated_output = list(output)
            changed_output = False
            for index, item in enumerate(output):
                if not isinstance(item, str):
                    continue
                updated_item, changed = _mask_completion_text(
                    span,
                    item,
                    span_name=span_name,
                    segment_index=index,
                )
                if changed:
                    updated_output[index] = updated_item
                    changed_output = True
            if changed_output:
                set_object_value(response, "output", updated_output)
            return response

        if not isinstance(output, str):
            return response
        updated_output, changed = _mask_completion_text(
            span,
            output,
            span_name=span_name,
            segment_index=0,
        )
        if changed:
            set_object_value(response, "output", updated_output)
        return response
    except Exception:
        return response


def _mask_prompt_value(span, value, *, span_name, segment_index, key=None):
    if isinstance(value, str):
        if not _should_mask_prompt_key(key):
            return value, False
        return _mask_prompt_text(
            span,
            value,
            span_name=span_name,
            segment_index=segment_index,
            segment_role="user",
        )

    if isinstance(value, dict):
        updated_value = value
        for child_key, child_value in value.items():
            updated_child, changed = _mask_prompt_value(
                span,
                child_value,
                span_name=span_name,
                segment_index=segment_index,
                key=child_key,
            )
            if not changed:
                continue
            if updated_value is value:
                updated_value = clone_value(value)
            updated_value[child_key] = updated_child
        return updated_value, updated_value is not value

    if not isinstance(value, list):
        return value, False

    updated_value = value
    for index, item in enumerate(value):
        updated_item, changed = _mask_prompt_value(
            span,
            item,
            span_name=span_name,
            segment_index=index if _should_mask_prompt_key(key) else segment_index,
            key=key,
        )
        if not changed:
            continue
        if updated_value is value:
            updated_value = clone_value(value)
        updated_value[index] = updated_item

    return updated_value, updated_value is not value


def _should_mask_prompt_key(key):
    if not isinstance(key, str):
        return False
    normalized = key.lower()
    return any(
        normalized == marker
        or normalized.startswith(f"{marker}_")
        or normalized.startswith(f"{marker}.")
        for marker in _PROMPT_KEY_MARKERS
    )


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
