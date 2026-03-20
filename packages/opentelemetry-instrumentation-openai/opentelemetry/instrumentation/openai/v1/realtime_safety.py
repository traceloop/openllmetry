from __future__ import annotations

from typing import Any

from opentelemetry.instrumentation.fortifyroot import (
    clone_value,
    get_object_value,
    set_object_value,
)
from opentelemetry.instrumentation.fortifyroot.text_streaming import (
    CompletionTextStreamGroup,
)
from opentelemetry.instrumentation.openai.shared.safety_common import (
    CHAT_PROVIDER,
    CHAT_SPAN_NAME,
    mask_prompt_text,
    request_type,
)


def apply_realtime_event_prompt_safety(event):
    if get_object_value(event, "type") != "conversation.item.create":
        return event

    item = get_object_value(event, "item")
    masked_item, changed = _mask_realtime_item(item)
    if not changed:
        return event

    updated_event = _clone_and_set(event, "item", masked_item)
    return updated_event if updated_event is not None else event


def apply_realtime_session_prompt_safety(kwargs: dict[str, Any]) -> dict[str, Any]:
    session = kwargs.get("session")
    instructions = get_object_value(session, "instructions")
    if not isinstance(instructions, str):
        return kwargs

    masked_instructions, changed = mask_prompt_text(
        None,
        instructions,
        span_name=CHAT_SPAN_NAME,
        segment_index=0,
        segment_role="system",
    )
    if not changed:
        return kwargs

    updated_session = _clone_and_set(session, "instructions", masked_instructions)
    if updated_session is None:
        return kwargs

    updated_kwargs = dict(kwargs)
    updated_kwargs["session"] = updated_session
    return updated_kwargs


def apply_realtime_item_prompt_safety(kwargs: dict[str, Any]) -> dict[str, Any]:
    item = kwargs.get("item")
    masked_item, changed = _mask_realtime_item(item)
    if not changed:
        return kwargs

    updated_kwargs = dict(kwargs)
    updated_kwargs["item"] = masked_item
    return updated_kwargs


def extract_realtime_input_text(item) -> str | None:
    content = get_object_value(item, "content")
    if not isinstance(content, list):
        return None

    for block in content:
        if get_object_value(block, "type") != "input_text":
            continue
        text = get_object_value(block, "text")
        if isinstance(text, str):
            return text

    return None


class RealtimeStreamingSafety:
    def __init__(self, span):
        self._text_streams = CompletionTextStreamGroup(
            span=span,
            provider=CHAT_PROVIDER,
            span_name=CHAT_SPAN_NAME,
            request_type=request_type(CHAT_SPAN_NAME),
        )
        self._transcript_streams = CompletionTextStreamGroup(
            span=span,
            provider=CHAT_PROVIDER,
            span_name=CHAT_SPAN_NAME,
            request_type=request_type(CHAT_SPAN_NAME),
        )
        self._done_tails: dict[str, tuple[str, str]] = {}

    def process_event(self, response_id: str | None, event):
        event_type = getattr(event, "type", None)
        response_key = self._response_key(response_id)

        if event_type == "response.text.delta":
            self._mask_delta(self._text_streams, ("text", response_key), event)
            return [event]

        if event_type == "response.audio_transcript.delta":
            self._mask_delta(
                self._transcript_streams,
                ("transcript", response_key),
                event,
            )
            return [event]

        if event_type == "response.done":
            self._done_tails[response_key] = (
                self._text_streams.flush(("text", response_key)),
                self._transcript_streams.flush(("transcript", response_key)),
            )
            return [event]

        return [event]

    def consume_done_tails(self, response_id: str | None) -> tuple[str, str]:
        return self._done_tails.pop(self._response_key(response_id), ("", ""))

    def _mask_delta(self, streams, key, event):
        delta = getattr(event, "delta", None)
        if not isinstance(delta, str):
            return None
        masked = streams.process(
            key,
            delta,
            segment_index=0,
            segment_role="assistant",
        )
        setattr(event, "delta", masked)
        return masked

    def _response_key(self, response_id: str | None) -> str:
        return response_id or "current"


def _mask_realtime_item(item):
    content = get_object_value(item, "content")
    if not isinstance(content, list):
        return item, False

    updated_content = content
    changed = False

    for block_index, block in enumerate(content):
        if get_object_value(block, "type") != "input_text":
            continue
        text = get_object_value(block, "text")
        if not isinstance(text, str):
            continue
        masked_text, text_changed = mask_prompt_text(
            None,
            text,
            span_name=CHAT_SPAN_NAME,
            segment_index=0,
            segment_role="user",
            metadata={"block_index": block_index},
        )
        if not text_changed:
            continue
        if updated_content is content:
            updated_content = clone_value(content)
        if not set_object_value(updated_content[block_index], "text", masked_text):
            continue
        changed = True

    if not changed:
        return item, False

    updated_item = _clone_and_set(item, "content", updated_content)
    if updated_item is None:
        return item, False
    return updated_item, True


def _clone_and_set(obj, key: str, value: Any):
    if obj is None:
        return None
    updated = clone_value(obj)
    if not set_object_value(updated, key, value):
        return None
    return updated
