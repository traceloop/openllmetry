from __future__ import annotations

import json
from io import BytesIO

from opentelemetry.instrumentation.fortifyroot import (
    SafetyDecision,
    SafetyLocation,
    run_completion_safety,
    run_prompt_safety,
)
from opentelemetry.instrumentation.bedrock.reusable_streaming_body import (
    ReusableStreamingBody,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues

PROVIDER = "Bedrock"


def _apply_invoke_prompt_safety(span, kwargs, span_name):
    payload, as_bytes = _decode_payload(kwargs.get("body"))
    if payload is None:
        return kwargs

    masked_payload, changed = _mask_prompt_payload(
        span,
        payload,
        span_name=span_name,
        request_type=_request_type(span_name),
        segment_index=0,
    )
    if not changed:
        return kwargs

    mutated_kwargs = dict(kwargs)
    mutated_kwargs["body"] = _encode_payload(masked_payload, as_bytes)
    return mutated_kwargs


def _apply_converse_prompt_safety(span, kwargs, span_name):
    try:
        mutated_kwargs = kwargs

        system_messages = kwargs.get("system")
        if isinstance(system_messages, list):
            mutated_system = None
            for index, message in enumerate(system_messages):
                text = message.get("text") if isinstance(message, dict) else None
                if not isinstance(text, str):
                    continue
                updated_text, changed = _mask_prompt_text(
                    span,
                    text,
                    span_name=span_name,
                    request_type=LLMRequestTypeValues.CHAT.value,
                    segment_index=index,
                    segment_role="system",
                )
                if not changed:
                    continue
                if mutated_system is None:
                    mutated_kwargs = dict(kwargs)
                    mutated_system = [dict(item) for item in system_messages]
                    mutated_kwargs["system"] = mutated_system
                mutated_system[index]["text"] = updated_text

        messages = kwargs.get("messages")
        if not isinstance(messages, list):
            return mutated_kwargs

        mutated_messages = None
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            updated_content, changed = _mask_prompt_content(
                span,
                message.get("content"),
                span_name=span_name,
                request_type=LLMRequestTypeValues.CHAT.value,
                segment_index=index,
                segment_role=message.get("role") or "user",
            )
            if not changed:
                continue
            if mutated_messages is None:
                if mutated_kwargs is kwargs:
                    mutated_kwargs = dict(kwargs)
                mutated_messages = [dict(item) if isinstance(item, dict) else item for item in messages]
                mutated_kwargs["messages"] = mutated_messages
            mutated_messages[index]["content"] = updated_content

        return mutated_kwargs
    except Exception:
        return kwargs


def _apply_invoke_completion_safety(span, raw_response, span_name):
    payload, as_bytes = _decode_payload(raw_response)
    if payload is None:
        return raw_response, False

    masked_payload, changed = _mask_completion_payload(
        span,
        payload,
        span_name=span_name,
        request_type=_request_type(span_name),
        segment_index=0,
    )
    if not changed:
        return raw_response, False
    return _encode_payload(masked_payload, as_bytes), True


def _prepare_invoke_response(span, response, span_name):
    body = response.get("body")
    if body is None:
        return None

    response["body"] = ReusableStreamingBody(body._raw_stream, body._content_length)
    raw_response = response["body"].read()
    masked_response, changed = _apply_invoke_completion_safety(
        span, raw_response, span_name
    )
    if changed:
        raw_response = masked_response
    response["body"] = ReusableStreamingBody(BytesIO(raw_response), len(raw_response))
    return json.loads(raw_response)


def _apply_converse_completion_safety(span, response, span_name):
    try:
        output = response.get("output")
        if not isinstance(output, dict):
            return
        message = output.get("message")
        if not isinstance(message, dict):
            return
        updated_content, changed = _mask_completion_content(
            span,
            message.get("content"),
            span_name=span_name,
            request_type=LLMRequestTypeValues.CHAT.value,
            segment_index=0,
        )
        if changed:
            updated_message = dict(message)
            updated_message["content"] = updated_content
            updated_output = dict(output)
            updated_output["message"] = updated_message
            response["output"] = updated_output
    except Exception:
        return


def _mask_prompt_payload(span, value, *, span_name, request_type, segment_index):
    if isinstance(value, dict):
        updated = value
        for key, item in value.items():
            if key in {"prompt", "inputText", "text"} and isinstance(item, str):
                updated_item, changed = _mask_prompt_text(
                    span,
                    item,
                    span_name=span_name,
                    request_type=request_type,
                    segment_index=segment_index,
                    segment_role="user",
                )
            elif key in {"messages", "content"}:
                updated_item, changed = _mask_prompt_content(
                    span,
                    item,
                    span_name=span_name,
                    request_type=request_type,
                    segment_index=segment_index,
                    segment_role="user",
                )
            else:
                updated_item, changed = _mask_prompt_payload(
                    span,
                    item,
                    span_name=span_name,
                    request_type=request_type,
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
                request_type=request_type,
                segment_index=index,
            )
            if not changed:
                continue
            if updated is value:
                updated = list(value)
            updated[index] = updated_item
        return updated, updated is not value

    return value, False


def _mask_completion_payload(span, value, *, span_name, request_type, segment_index):
    if isinstance(value, dict):
        updated = value
        for key, item in value.items():
            if key in {"completion", "outputText", "generated_text", "text"} and isinstance(item, str):
                updated_item, changed = _mask_completion_text(
                    span,
                    item,
                    span_name=span_name,
                    request_type=request_type,
                    segment_index=segment_index,
                )
            elif key in {"content", "completions", "generations"}:
                updated_item, changed = _mask_completion_content(
                    span,
                    item,
                    span_name=span_name,
                    request_type=request_type,
                    segment_index=segment_index,
                )
            else:
                updated_item, changed = _mask_completion_payload(
                    span,
                    item,
                    span_name=span_name,
                    request_type=request_type,
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
                request_type=request_type,
                segment_index=index,
            )
            if not changed:
                continue
            if updated is value:
                updated = list(value)
            updated[index] = updated_item
        return updated, updated is not value

    return value, False


def _mask_prompt_content(
    span,
    content,
    *,
    span_name,
    request_type,
    segment_index,
    segment_role,
):
    if isinstance(content, str):
        return _mask_prompt_text(
            span,
            content,
            span_name=span_name,
            request_type=request_type,
            segment_index=segment_index,
            segment_role=segment_role,
        )

    if not isinstance(content, list):
        return content, False

    updated = content
    for index, item in enumerate(content):
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            updated_text, changed = _mask_prompt_text(
                span,
                item["text"],
                span_name=span_name,
                request_type=request_type,
                segment_index=segment_index,
                segment_role=segment_role,
            )
            if not changed:
                continue
            if updated is content:
                updated = [dict(block) if isinstance(block, dict) else block for block in content]
            updated[index]["text"] = updated_text
            continue

        updated_item, changed = _mask_prompt_payload(
            span,
            item,
            span_name=span_name,
            request_type=request_type,
            segment_index=index,
        )
        if not changed:
            continue
        if updated is content:
            updated = list(content)
        updated[index] = updated_item

    return updated, updated is not content


def _mask_completion_content(span, content, *, span_name, request_type, segment_index):
    if isinstance(content, str):
        return _mask_completion_text(
            span,
            content,
            span_name=span_name,
            request_type=request_type,
            segment_index=segment_index,
        )

    if not isinstance(content, list):
        return content, False

    updated = content
    for index, item in enumerate(content):
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            updated_text, changed = _mask_completion_text(
                span,
                item["text"],
                span_name=span_name,
                request_type=request_type,
                segment_index=segment_index,
            )
            if not changed:
                continue
            if updated is content:
                updated = [dict(block) if isinstance(block, dict) else block for block in content]
            updated[index]["text"] = updated_text
            continue

        updated_item, changed = _mask_completion_payload(
            span,
            item,
            span_name=span_name,
            request_type=request_type,
            segment_index=index,
        )
        if not changed:
            continue
        if updated is content:
            updated = list(content)
        updated[index] = updated_item

    return updated, updated is not content


def _mask_prompt_text(
    span,
    text,
    *,
    span_name,
    request_type,
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


def _mask_completion_text(span, text, *, span_name, request_type, segment_index):
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


def _request_type(span_name):
    if "converse" in span_name:
        return LLMRequestTypeValues.CHAT.value
    return LLMRequestTypeValues.COMPLETION.value


def _resolve_masked_text(original_text, result):
    if result is None or result.overall_action != SafetyDecision.MASK.value:
        return original_text, False
    if result.text == original_text:
        return original_text, False
    return result.text, True
