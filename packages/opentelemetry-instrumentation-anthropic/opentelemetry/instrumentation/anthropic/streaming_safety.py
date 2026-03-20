from __future__ import annotations

from opentelemetry.instrumentation.anthropic.safety import PROVIDER, _request_type
from opentelemetry.instrumentation.fortifyroot import get_object_value, set_object_value
from opentelemetry.instrumentation.fortifyroot.text_streaming import (
    CompletionTextStreamGroup,
)


_DELTA_ROLE_BY_TYPE = {
    "text_delta": "assistant",
    "thinking_delta": "thinking",
}


class AnthropicStreamingSafety:
    def __init__(self, span, span_name: str):
        self._streams = CompletionTextStreamGroup(
            span=span,
            provider=PROVIDER,
            span_name=span_name,
            request_type=_request_type(span_name),
        )

    def process_item(self, item):
        if get_object_value(item, "type") != "content_block_delta":
            return item

        delta = get_object_value(item, "delta")
        delta_type = get_object_value(delta, "type")
        role = _DELTA_ROLE_BY_TYPE.get(delta_type)
        if role is None:
            return item

        text_key = "text" if delta_type == "text_delta" else "thinking"
        text = get_object_value(delta, text_key)
        if not isinstance(text, str):
            return item

        masked = self._streams.process(
            (get_object_value(item, "index", 0) or 0, role),
            text,
            segment_index=get_object_value(item, "index", 0) or 0,
            segment_role=role,
        )
        set_object_value(delta, text_key, masked)
        return item

    def flush_transition(self, pending_item, current_item):
        key = self._flush_key(pending_item, current_item)
        if key is None:
            return
        tail = self._streams.flush(key)
        if tail:
            self._append_tail(pending_item, key, tail)

    def flush_pending_item(self, pending_item):
        if pending_item is None:
            return
        key = self._delta_key(pending_item)
        if key is None:
            return
        tail = self._streams.flush(key)
        if tail:
            self._append_tail(pending_item, key, tail)

    def _flush_key(self, pending_item, current_item):
        if self._delta_key(pending_item) is None:
            return None
        item_type = get_object_value(current_item, "type")
        if item_type in ("content_block_stop", "message_delta", "message_stop"):
            return self._delta_key(pending_item)
        return None

    def _delta_key(self, item):
        if get_object_value(item, "type") != "content_block_delta":
            return None
        delta = get_object_value(item, "delta")
        delta_type = get_object_value(delta, "type")
        role = _DELTA_ROLE_BY_TYPE.get(delta_type)
        if role is None:
            return None
        return (get_object_value(item, "index", 0) or 0, role)

    def _append_tail(self, item, key, tail: str):
        if self._delta_key(item) != key:
            return
        delta = get_object_value(item, "delta")
        delta_type = get_object_value(delta, "type")
        text_key = "text" if delta_type == "text_delta" else "thinking"
        current_text = get_object_value(delta, text_key) or ""
        set_object_value(delta, text_key, f"{current_text}{tail}")
