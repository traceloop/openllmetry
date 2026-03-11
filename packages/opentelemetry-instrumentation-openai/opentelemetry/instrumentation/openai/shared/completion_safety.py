from __future__ import annotations

from opentelemetry.instrumentation.openai.shared.safety_common import (
    COMPLETION_SPAN_NAME,
    mask_completion_text,
    mask_prompt_text,
)
from opentelemetry.instrumentation.fortifyroot import get_object_value, set_object_value


def _apply_prompt_safety(span, kwargs):
    try:
        prompt = kwargs.get("prompt")
        if isinstance(prompt, str):
            updated_prompt, changed = mask_prompt_text(
                span,
                prompt,
                span_name=COMPLETION_SPAN_NAME,
                segment_index=0,
                segment_role="user",
            )
            if not changed:
                return kwargs
            mutated_kwargs = dict(kwargs)
            mutated_kwargs["prompt"] = updated_prompt
            return mutated_kwargs

        if not isinstance(prompt, list):
            return kwargs

        mutated_prompt = None
        for index, prompt_text in enumerate(prompt):
            if not isinstance(prompt_text, str):
                continue
            updated_prompt, changed = mask_prompt_text(
                span,
                prompt_text,
                span_name=COMPLETION_SPAN_NAME,
                segment_index=index,
                segment_role="user",
            )
            if not changed:
                continue
            if mutated_prompt is None:
                mutated_prompt = list(prompt)
            mutated_prompt[index] = updated_prompt
        if mutated_prompt is None:
            return kwargs
        mutated_kwargs = dict(kwargs)
        mutated_kwargs["prompt"] = mutated_prompt
        return mutated_kwargs
    except Exception:
        return kwargs


def _apply_completion_safety(span, response):
    try:
        for index, choice in enumerate(get_object_value(response, "choices", []) or []):
            text = get_object_value(choice, "text")
            if not isinstance(text, str):
                continue
            updated_text, changed = mask_completion_text(
                span,
                text,
                span_name=COMPLETION_SPAN_NAME,
                segment_index=index,
            )
            if changed:
                set_object_value(choice, "text", updated_text)
    except Exception:
        return
