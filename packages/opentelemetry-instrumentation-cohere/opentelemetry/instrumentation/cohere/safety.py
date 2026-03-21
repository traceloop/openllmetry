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

PROVIDER = "Cohere"


def _apply_prompt_safety(span, kwargs, llm_request_type, span_name):
    try:
        if llm_request_type not in (
            LLMRequestTypeValues.CHAT,
            LLMRequestTypeValues.COMPLETION,
        ):
            return kwargs

        mutated_kwargs = kwargs

        if llm_request_type == LLMRequestTypeValues.COMPLETION:
            prompt = kwargs.get("prompt")
            if not isinstance(prompt, str):
                return kwargs
            updated_prompt, changed = _mask_prompt_text(
                span,
                prompt,
                request_type=llm_request_type.value,
                span_name=span_name,
                segment_index=0,
                segment_role="user",
            )
            if not changed:
                return kwargs
            mutated_kwargs = dict(kwargs)
            mutated_kwargs["prompt"] = updated_prompt
            return mutated_kwargs

        preamble = kwargs.get("preamble")
        if isinstance(preamble, str):
            updated_preamble, changed = _mask_prompt_text(
                span,
                preamble,
                request_type=llm_request_type.value,
                span_name=span_name,
                segment_index=0,
                segment_role="system",
            )
            if changed:
                mutated_kwargs = dict(kwargs)
                mutated_kwargs["preamble"] = updated_preamble

        message = kwargs.get("message")
        if isinstance(message, str):
            updated_message, changed = _mask_prompt_text(
                span,
                message,
                request_type=llm_request_type.value,
                span_name=span_name,
                segment_index=0,
                segment_role="user",
            )
            if changed:
                if mutated_kwargs is kwargs:
                    mutated_kwargs = dict(kwargs)
                mutated_kwargs["message"] = updated_message

        messages = kwargs.get("messages")
        if not isinstance(messages, list):
            return mutated_kwargs

        mutated_messages = None
        for index, message_obj in enumerate(messages):
            role = get_object_value(message_obj, "role")
            content = get_object_value(message_obj, "content")
            updated_content, changed = _mask_prompt_content(
                span,
                content,
                request_type=llm_request_type.value,
                span_name=span_name,
                segment_index=index,
                segment_role=role,
            )
            if not changed:
                continue
            if mutated_messages is None:
                if mutated_kwargs is kwargs:
                    mutated_kwargs = dict(kwargs)
                mutated_messages = clone_value(messages)
                mutated_kwargs["messages"] = mutated_messages
            set_object_value(mutated_messages[index], "content", updated_content)

        return mutated_kwargs
    except Exception:
        return kwargs


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
    return content, False


def _apply_completion_safety(span, response, llm_request_type, span_name):
    try:
        if llm_request_type not in (
            LLMRequestTypeValues.CHAT,
            LLMRequestTypeValues.COMPLETION,
        ):
            return

        if llm_request_type == LLMRequestTypeValues.CHAT:
            text = get_object_value(response, "text")
            if isinstance(text, str):
                updated_text, changed = _mask_completion_text(
                    span,
                    text,
                    request_type=llm_request_type.value,
                    span_name=span_name,
                    segment_index=0,
                )
                if changed:
                    set_object_value(response, "text", updated_text)

            message = get_object_value(response, "message")
            if message is not None:
                content = get_object_value(message, "content")
                original_content_text = _cohere_content_text(content)
                updated_content, changed = _mask_completion_content(
                    span,
                    content,
                    request_type=llm_request_type.value,
                    span_name=span_name,
                    segment_index=0,
                )
                if changed:
                    set_object_value(message, "content", updated_content)
                    updated_response_text = _cohere_content_text(updated_content)
                    if isinstance(updated_response_text, str) and (
                        text is None or text == original_content_text
                    ):
                        set_object_value(response, "text", updated_response_text)
            return

        generations = get_object_value(response, "generations")
        if generations is None:
            generations = response

        for index, generation in enumerate(generations or []):
            text = get_object_value(generation, "text")
            if not isinstance(text, str):
                continue
            updated_text, changed = _mask_completion_text(
                span,
                text,
                request_type=llm_request_type.value,
                span_name=span_name,
                segment_index=index,
            )
            if changed:
                set_object_value(generation, "text", updated_text)
    except Exception:
        return


def _mask_completion_content(
    span,
    content,
    *,
    request_type,
    span_name,
    segment_index,
):
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
    for index, block in enumerate(content):
        text = get_object_value(block, "text")
        if not isinstance(text, str):
            continue
        updated_text, changed = _mask_completion_text(
            span,
            text,
            request_type=request_type,
            span_name=span_name,
            segment_index=index,
        )
        if not changed:
            continue
        if updated_content is content:
            updated_content = clone_value(content)
        set_object_value(updated_content[index], "text", updated_text)

    return updated_content, updated_content is not content


def _mask_prompt_text(
    span,
    text,
    *,
    request_type,
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
        request_type=request_type,
        segment_index=segment_index,
        segment_role=segment_role,
    )
    return _resolve_masked_text(text, result)


def _mask_completion_text(
    span,
    text,
    *,
    request_type,
    span_name,
    segment_index,
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
    )
    return _resolve_masked_text(text, result)


def _resolve_masked_text(original_text, result):
    if result is None or result.overall_action != SafetyDecision.MASK.value:
        return original_text, False
    if result.text == original_text:
        return original_text, False
    return result.text, True


def _cohere_content_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts = []
    for block in content:
        text = get_object_value(block, "text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)
