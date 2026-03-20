from __future__ import annotations

from opentelemetry.instrumentation.fortifyroot import get_object_value, set_object_value
from opentelemetry.instrumentation.fortifyroot.text_streaming import (
    CompletionTextStreamGroup,
)
from opentelemetry.instrumentation.groq.safety import PROVIDER
from opentelemetry.semconv_ai import LLMRequestTypeValues


class GroqStreamingSafety:
    def __init__(self, span, span_name: str):
        self._streams = CompletionTextStreamGroup(
            span=span,
            provider=PROVIDER,
            span_name=span_name,
            request_type=LLMRequestTypeValues.CHAT.value,
        )

    def process_chunk(self, chunk):
        for index, choice in enumerate(get_object_value(chunk, "choices") or []):
            delta = get_object_value(choice, "delta")
            content = get_object_value(delta, "content")
            if isinstance(content, str):
                masked = self._streams.process(
                    ("choice", index),
                    content,
                    segment_index=index,
                    segment_role="assistant",
                )
                set_object_value(delta, "content", masked)
            if get_object_value(choice, "finish_reason"):
                tail = self._streams.flush(("choice", index))
                if tail:
                    current_text = get_object_value(delta, "content") or ""
                    set_object_value(delta, "content", f"{current_text}{tail}")
        return chunk
