from __future__ import annotations

from opentelemetry.instrumentation.cohere.safety import PROVIDER
from opentelemetry.instrumentation.fortifyroot import (
    get_object_value,
    set_object_value,
)
from opentelemetry.instrumentation.fortifyroot.text_streaming import (
    CompletionTextStreamGroup,
)


class CohereStreamingSafety:
    def __init__(self, span, span_name: str, request_type: str):
        self._streams = CompletionTextStreamGroup(
            span=span,
            provider=PROVIDER,
            span_name=span_name,
            request_type=request_type,
        )

    def process_v1_item(self, item):
        if get_object_value(item, "event_type") != "text-generation":
            return item

        text = get_object_value(item, "text")
        if not isinstance(text, str):
            return item

        masked = self._streams.process(
            ("text", 0),
            text,
            segment_index=0,
            segment_role="assistant",
        )
        set_object_value(item, "text", masked)
        return item

    def process_v2_item(self, item):
        if get_object_value(item, "type") != "content-delta":
            return item

        content = _content_delta(item)
        if content is None:
            return item

        block_index = get_object_value(item, "index", 0) or 0
        for field_name, role in (("text", "assistant"), ("thinking", "thinking")):
            text = get_object_value(content, field_name)
            if not isinstance(text, str):
                continue
            masked = self._streams.process(
                (block_index, role),
                text,
                segment_index=block_index,
                segment_role=role,
            )
            set_object_value(content, field_name, masked)

        return item

    def flush_transition(self, pending_item, current_item):
        key = self._flush_key(pending_item, current_item)
        if key is None:
            return
        tail = self._streams.flush(key)
        if tail:
            self._append_tail(pending_item, key, tail)

    def flush_pending_item(self, pending_item):
        key = self._delta_key(pending_item)
        if key is None:
            return
        tail = self._streams.flush(key)
        if tail:
            self._append_tail(pending_item, key, tail)

    def _flush_key(self, pending_item, current_item):
        key = self._delta_key(pending_item)
        if key is None:
            return None

        if get_object_value(pending_item, "event_type") == "text-generation":
            if get_object_value(current_item, "event_type") == "stream-end":
                return key

        current_type = get_object_value(current_item, "type")
        if current_type in ("content-end", "message-end"):
            return key
        return None

    def _delta_key(self, item):
        event_type = get_object_value(item, "event_type")
        if event_type == "text-generation":
            return ("text", 0)

        if get_object_value(item, "type") != "content-delta":
            return None

        content = _content_delta(item)
        if content is None:
            return None

        block_index = get_object_value(item, "index", 0) or 0
        if isinstance(get_object_value(content, "thinking"), str):
            return (block_index, "thinking")
        if isinstance(get_object_value(content, "text"), str):
            return (block_index, "assistant")
        return None

    def _append_tail(self, item, key, tail: str):
        if get_object_value(item, "event_type") == "text-generation":
            current_text = get_object_value(item, "text") or ""
            set_object_value(item, "text", f"{current_text}{tail}")
            return

        content = _content_delta(item)
        if content is None:
            return

        _, role = key
        field_name = "thinking" if role == "thinking" else "text"
        current_text = get_object_value(content, field_name) or ""
        set_object_value(content, field_name, f"{current_text}{tail}")


def _content_delta(item):
    delta = get_object_value(item, "delta")
    if delta is None:
        return None
    message = get_object_value(delta, "message")
    if message is None:
        return None
    return get_object_value(message, "content")
