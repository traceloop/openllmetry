from __future__ import annotations

import json

from opentelemetry.instrumentation.fortifyroot import (
    SafetyDecision,
    SafetyLocation,
    run_completion_safety,
    run_prompt_safety,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues

PROVIDER = "SageMaker"
_PROMPT_KEYS = {"prompt", "inputs", "inputText", "input_text", "text"}
_COMPLETION_KEYS = {"generated_text", "text", "completion", "output", "response"}


def _apply_prompt_safety(span, kwargs, span_name):
    try:
        body = kwargs.get("Body")
        payload, as_bytes = _decode_payload(body)
        if payload is None:
            return kwargs

        masked_payload, changed = _mask_prompt_payload(
            span,
            payload,
            span_name=span_name,
            segment_index=0,
        )
        if not changed:
            return kwargs

        mutated_kwargs = dict(kwargs)
        mutated_kwargs["Body"] = _encode_payload(masked_payload, as_bytes)
        return mutated_kwargs
    except Exception:
        return kwargs


def _apply_completion_safety(span, raw_response, span_name):
    try:
        payload, as_bytes = _decode_payload(raw_response)
        if payload is None:
            return raw_response, False

        masked_payload, changed = _mask_completion_payload(
            span,
            payload,
            span_name=span_name,
            segment_index=0,
        )
        if not changed:
            return raw_response, False

        return _encode_payload(masked_payload, as_bytes), True
    except Exception:
        return raw_response, False


def _mask_prompt_payload(span, value, *, span_name, segment_index):
    if isinstance(value, dict):
        updated = value
        for key, item in value.items():
            if key in _PROMPT_KEYS and isinstance(item, str):
                updated_item, changed = _mask_prompt_text(
                    span,
                    item,
                    span_name=span_name,
                    segment_index=segment_index,
                    segment_role="user",
                )
            else:
                updated_item, changed = _mask_prompt_payload(
                    span,
                    item,
                    span_name=span_name,
                    segment_index=segment_index,
                )
            if not changed:
                continue
            if updated is value:
                updated = dict(value)
            updated[key] = updated_item
        return updated, updated is not value

    if isinstance(value, list):
        updated = value
        for index, item in enumerate(value):
            updated_item, changed = _mask_prompt_payload(
                span,
                item,
                span_name=span_name,
                segment_index=index,
            )
            if not changed:
                continue
            if updated is value:
                updated = list(value)
            updated[index] = updated_item
        return updated, updated is not value

    return value, False


def _mask_completion_payload(span, value, *, span_name, segment_index):
    if isinstance(value, dict):
        updated = value
        for key, item in value.items():
            if key in _COMPLETION_KEYS and isinstance(item, str):
                updated_item, changed = _mask_completion_text(
                    span,
                    item,
                    span_name=span_name,
                    segment_index=segment_index,
                )
            else:
                updated_item, changed = _mask_completion_payload(
                    span,
                    item,
                    span_name=span_name,
                    segment_index=segment_index,
                )
            if not changed:
                continue
            if updated is value:
                updated = dict(value)
            updated[key] = updated_item
        return updated, updated is not value

    if isinstance(value, list):
        updated = value
        for index, item in enumerate(value):
            updated_item, changed = _mask_completion_payload(
                span,
                item,
                span_name=span_name,
                segment_index=index,
            )
            if not changed:
                continue
            if updated is value:
                updated = list(value)
            updated[index] = updated_item
        return updated, updated is not value

    return value, False


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


def _decode_payload(raw_value):
    try:
        if isinstance(raw_value, bytes):
            return json.loads(raw_value.decode("utf-8")), True
        if isinstance(raw_value, str):
            return json.loads(raw_value), False
    except Exception:
        return None, False
    return None, False


def _encode_payload(payload, as_bytes):
    encoded = json.dumps(payload).encode("utf-8")
    if as_bytes:
        return encoded
    return encoded.decode("utf-8")


def _resolve_masked_text(original_text, result):
    if result is None or result.overall_action != SafetyDecision.MASK.value:
        return original_text, False
    if result.text == original_text:
        return original_text, False
    return result.text, True
