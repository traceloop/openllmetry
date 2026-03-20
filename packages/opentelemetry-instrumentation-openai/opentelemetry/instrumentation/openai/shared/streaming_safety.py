from __future__ import annotations

from collections.abc import MutableMapping
from types import SimpleNamespace

from opentelemetry.instrumentation.fortifyroot import (
    get_object_value,
    set_object_value,
)
from opentelemetry.instrumentation.fortifyroot.text_streaming import (
    CompletionTextStreamGroup,
)
from opentelemetry.instrumentation.openai.shared.safety_common import (
    CHAT_PROVIDER,
    request_type,
)


class OpenAIChatStreamingSafety:
    def __init__(self, span, span_name: str):
        self._streams = CompletionTextStreamGroup(
            span=span,
            provider=CHAT_PROVIDER,
            span_name=span_name,
            request_type=request_type(span_name),
        )

    def process_chunk(self, item):
        for choice in get_object_value(item, "choices", []) or []:
            index = get_object_value(choice, "index", 0) or 0
            delta = _ensure_delta(choice)
            if delta is None:
                continue
            content = get_object_value(delta, "content")
            if isinstance(content, str):
                masked = self._streams.process(
                    ("choice", index),
                    content,
                    segment_index=index,
                    segment_role="assistant",
                )
                set_object_value(delta, "content", masked)

            finish_reason = get_object_value(choice, "finish_reason")
            if finish_reason:
                tail = self._streams.flush(("choice", index))
                if tail:
                    current_text = get_object_value(delta, "content") or ""
                    set_object_value(delta, "content", f"{current_text}{tail}")
        return item


class OpenAICompletionStreamingSafety:
    def __init__(self, span, span_name: str):
        self._streams = CompletionTextStreamGroup(
            span=span,
            provider=CHAT_PROVIDER,
            span_name=span_name,
            request_type=request_type(span_name),
        )

    def process_chunk(self, item):
        for choice in get_object_value(item, "choices", []) or []:
            index = get_object_value(choice, "index", 0) or 0
            text = get_object_value(choice, "text")
            if isinstance(text, str):
                masked = self._streams.process(
                    ("choice", index),
                    text,
                    segment_index=index,
                    segment_role="assistant",
                )
                set_object_value(choice, "text", masked)

            finish_reason = get_object_value(choice, "finish_reason")
            if finish_reason:
                tail = self._streams.flush(("choice", index))
                if tail:
                    current_text = get_object_value(choice, "text") or ""
                    set_object_value(choice, "text", f"{current_text}{tail}")
        return item


def _ensure_delta(choice):
    delta = get_object_value(choice, "delta")
    if delta is not None:
        return delta

    if isinstance(choice, MutableMapping):
        delta = {}
        choice["delta"] = delta
        return delta

    delta = SimpleNamespace()
    if set_object_value(choice, "delta", delta):
        return delta
    return None
