from __future__ import annotations

from opentelemetry.instrumentation.fortifyroot import (
    SafetyDecision,
    SafetyLocation,
    run_completion_safety,
    run_prompt_safety,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues

PROVIDER = "Watsonx"


def _apply_prompt_safety(span, args, kwargs, span_name):
    try:
        prompt = kwargs.get("prompt")
        if prompt is None and args:
            first_arg = args[0]
            if isinstance(first_arg, (str, list)):
                prompt = first_arg

        if isinstance(prompt, str):
            updated_prompt, changed = _mask_prompt_text(
                span,
                prompt,
                span_name=span_name,
                segment_index=0,
                segment_role="user",
            )
            if not changed:
                return args, kwargs
            if "prompt" in kwargs:
                mutated_kwargs = dict(kwargs)
                mutated_kwargs["prompt"] = updated_prompt
                return args, mutated_kwargs
            return (updated_prompt, *args[1:]), kwargs

        if not isinstance(prompt, list):
            return args, kwargs

        updated_prompt = None
        for index, item in enumerate(prompt):
            if not isinstance(item, str):
                continue
            masked_item, changed = _mask_prompt_text(
                span,
                item,
                span_name=span_name,
                segment_index=index,
                segment_role="user",
            )
            if not changed:
                continue
            if updated_prompt is None:
                updated_prompt = list(prompt)
            updated_prompt[index] = masked_item

        if updated_prompt is None:
            return args, kwargs
        if "prompt" in kwargs:
            mutated_kwargs = dict(kwargs)
            mutated_kwargs["prompt"] = updated_prompt
            return args, mutated_kwargs
        return (updated_prompt, *args[1:]), kwargs
    except Exception:
        return args, kwargs


def _apply_completion_safety(span, responses, span_name):
    try:
        if isinstance(responses, list):
            for index, response in enumerate(responses):
                _apply_completion_to_response(span, response, span_name, index)
            return
        _apply_completion_to_response(span, responses, span_name, 0)
    except Exception:
        return


def _apply_completion_to_response(span, response, span_name, segment_index):
    if not isinstance(response, dict):
        return
    results = response.get("results")
    if not isinstance(results, list) or not results:
        return

    updated_results = None
    for result_index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        generated_text = result.get("generated_text")
        if not isinstance(generated_text, str):
            continue
        updated_text, changed = _mask_completion_text(
            span,
            generated_text,
            span_name=span_name,
            segment_index=result_index,
        )
        if not changed:
            continue
        if updated_results is None:
            updated_results = list(results)
        updated_result = dict(updated_results[result_index])
        updated_result["generated_text"] = updated_text
        updated_results[result_index] = updated_result

    if updated_results is not None:
        response["results"] = updated_results


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
