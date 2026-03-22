from __future__ import annotations

import json
import logging
from io import BytesIO

from opentelemetry.instrumentation.fortifyroot import (
    SafetyDecision,
    SafetyLocation,
    clone_value,
    get_object_value,
    run_completion_safety,
    run_prompt_safety,
    set_object_value,
)
from opentelemetry.instrumentation.bedrock.reusable_streaming_body import (
    ReusableStreamingBody,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues

PROVIDER = "Bedrock"
logger = logging.getLogger(__name__)


def _apply_invoke_prompt_safety(span, kwargs, span_name):
    try:
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
    except Exception:
        logger.warning("Prompt safety handling failed", exc_info=True)
        return kwargs


def _apply_converse_prompt_safety(span, kwargs, span_name):
    try:
        mutated_kwargs = kwargs

        system_messages = get_object_value(kwargs, "system")
        if isinstance(system_messages, list):
            mutated_system = None
            for index, message in enumerate(system_messages):
                text = get_object_value(message, "text") if message is not None else None
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
                    mutated_system = clone_value(system_messages)
                    mutated_kwargs["system"] = mutated_system
                set_object_value(mutated_system[index], "text", updated_text)

        messages = get_object_value(kwargs, "messages")
        if not isinstance(messages, list):
            return mutated_kwargs

        mutated_messages = None
        for index, message in enumerate(messages):
            if message is None:
                continue
            updated_content, changed = _mask_prompt_content(
                span,
                get_object_value(message, "content"),
                span_name=span_name,
                request_type=LLMRequestTypeValues.CHAT.value,
                segment_index=index,
                segment_role=get_object_value(message, "role") or "user",
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


def _apply_invoke_completion_safety(span, raw_response, span_name):
    try:
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
    except Exception:
        logger.warning("Completion safety handling failed", exc_info=True)
        return raw_response, False


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
        output = get_object_value(response, "output")
        if output is None:
            return
        message = get_object_value(output, "message")
        if message is None:
            return
        updated_content, changed = _mask_completion_content(
            span,
            get_object_value(message, "content"),
            span_name=span_name,
            request_type=LLMRequestTypeValues.CHAT.value,
            segment_index=0,
        )
        if changed:
            updated_message = clone_value(message)
            set_object_value(updated_message, "content", updated_content)
            updated_output = clone_value(output)
            set_object_value(updated_output, "message", updated_message)
            set_object_value(response, "output", updated_output)
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
                updated = clone_value(value)
            set_object_value(updated, key, updated_item)
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
                updated = clone_value(value)
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
                updated = clone_value(value)
            set_object_value(updated, key, updated_item)
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
                updated = clone_value(value)
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
        block_text = get_object_value(item, "text") if item is not None else None
        if isinstance(block_text, str):
            updated_text, changed = _mask_prompt_text(
                span,
                block_text,
                span_name=span_name,
                request_type=request_type,
                segment_index=segment_index,
                segment_role=segment_role,
            )
            if not changed:
                continue
            if updated is content:
                updated = clone_value(content)
            set_object_value(updated[index], "text", updated_text)
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
            updated = clone_value(content)
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
        block_text = get_object_value(item, "text") if item is not None else None
        if isinstance(block_text, str):
            updated_text, changed = _mask_completion_text(
                span,
                block_text,
                span_name=span_name,
                request_type=request_type,
                segment_index=segment_index,
            )
            if not changed:
                continue
            if updated is content:
                updated = clone_value(content)
            set_object_value(updated[index], "text", updated_text)
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
            updated = clone_value(content)
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
