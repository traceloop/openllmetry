from __future__ import annotations

from opentelemetry.instrumentation.fortifyroot import (
    SafetyDecision,
    SafetyLocation,
    run_completion_safety,
    run_prompt_safety,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues

CHAT_PROVIDER = "OpenAI"
CHAT_SPAN_NAME = "openai.chat"
COMPLETION_SPAN_NAME = "openai.completion"


def mask_prompt_text(
    span,
    text,
    *,
    span_name,
    segment_index,
    segment_role=None,
    metadata=None,
):
    result = run_prompt_safety(
        span=span,
        provider=CHAT_PROVIDER,
        span_name=span_name,
        text=text,
        location=SafetyLocation.PROMPT,
        request_type=request_type(span_name),
        segment_index=segment_index,
        segment_role=segment_role,
        metadata=metadata,
    )
    return resolve_masked_text(text, result)


def mask_completion_text(
    span,
    text,
    *,
    span_name,
    segment_index,
    metadata=None,
):
    result = run_completion_safety(
        span=span,
        provider=CHAT_PROVIDER,
        span_name=span_name,
        text=text,
        location=SafetyLocation.COMPLETION,
        request_type=request_type(span_name),
        segment_index=segment_index,
        segment_role="assistant",
        metadata=metadata,
    )
    return resolve_masked_text(text, result)


def request_type(span_name: str) -> str:
    if span_name == COMPLETION_SPAN_NAME:
        return LLMRequestTypeValues.COMPLETION.value
    return LLMRequestTypeValues.CHAT.value


def resolve_masked_text(original_text, result):
    if result is None or result.overall_action != SafetyDecision.MASK.value:
        return original_text, False
    if result.text == original_text:
        return original_text, False
    return result.text, True
