from __future__ import annotations

try:
    from aleph_alpha_client import Prompt as AlephAlphaPrompt
except Exception:  # pragma: no cover
    AlephAlphaPrompt = None
from opentelemetry.instrumentation.fortifyroot import (
    SafetyDecision,
    SafetyLocation,
    get_object_value,
    run_completion_safety,
    run_prompt_safety,
    set_object_value,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues

PROVIDER = "AlephAlpha"


def _apply_prompt_safety(span, args, kwargs, span_name):
    try:
        request = kwargs.get("request") or (args[0] if args else None)
        prompt = get_object_value(request, "prompt")
        updated_prompt, changed = _mask_prompt_value(
            span,
            prompt,
            span_name=span_name,
        )
        if not changed:
            return args, kwargs

        set_object_value(request, "prompt", updated_prompt)
        return args, kwargs
    except Exception:
        return args, kwargs


def _apply_completion_safety(span, response, span_name):
    try:
        completions = get_object_value(response, "completions")
        if not completions:
            return
        for index, completion_item in enumerate(completions):
            completion = get_object_value(completion_item, "completion")
            if not isinstance(completion, str):
                continue
            updated_completion, changed = _mask_completion_text(
                span,
                completion,
                span_name=span_name,
                segment_index=index,
            )
            if changed:
                set_object_value(completion_item, "completion", updated_completion)
    except Exception:
        return


def _mask_prompt_value(span, prompt, *, span_name):
    if isinstance(prompt, str):
        updated_text, changed = _mask_prompt_text(
            span,
            prompt,
            span_name=span_name,
            segment_index=0,
            segment_role="user",
        )
        if not changed:
            return prompt, False
        return updated_text, True

    if prompt is None:
        return prompt, False

    prompt_json = prompt.to_json() if hasattr(prompt, "to_json") else prompt
    updated_json, changed = _mask_prompt_json(span, prompt_json, span_name=span_name)
    if not changed:
        return prompt, False

    if isinstance(updated_json, list):
        if AlephAlphaPrompt is not None:
            return AlephAlphaPrompt.from_json(updated_json), True
        return updated_json, True
    return updated_json, True


def _mask_prompt_json(span, prompt_json, *, span_name):
    if isinstance(prompt_json, list):
        updated_prompt = prompt_json
        for index, item in enumerate(prompt_json):
            updated_item, changed = _mask_prompt_item(
                span,
                item,
                span_name=span_name,
                segment_index=index,
            )
            if not changed:
                continue
            if updated_prompt is prompt_json:
                updated_prompt = list(prompt_json)
            updated_prompt[index] = updated_item
        return updated_prompt, updated_prompt is not prompt_json

    if isinstance(prompt_json, dict):
        return _mask_prompt_item(
            span,
            prompt_json,
            span_name=span_name,
            segment_index=0,
        )

    return prompt_json, False


def _mask_prompt_item(span, item, *, span_name, segment_index):
    if not isinstance(item, dict):
        return item, False

    text_key = None
    if item.get("type") in (None, "text") and isinstance(item.get("data"), str):
        text_key = "data"
    elif item.get("type") in (None, "text") and isinstance(item.get("text"), str):
        text_key = "text"

    if text_key is None:
        return item, False

    updated_text, changed = _mask_prompt_text(
        span,
        item[text_key],
        span_name=span_name,
        segment_index=segment_index,
        segment_role="user",
    )
    if not changed:
        return item, False

    updated_item = dict(item)
    updated_item[text_key] = updated_text
    return updated_item, True


def _mask_prompt_text(
    span,
    text,
    *,
    span_name,
    segment_index,
    segment_role,
):
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
