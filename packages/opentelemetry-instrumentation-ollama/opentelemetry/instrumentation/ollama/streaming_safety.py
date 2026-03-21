from __future__ import annotations

from opentelemetry.instrumentation.fortifyroot import (
    get_object_value,
    set_object_value,
)
from opentelemetry.instrumentation.fortifyroot.text_streaming import (
    CompletionTextStreamGroup,
)
from opentelemetry.instrumentation.ollama.safety import PROVIDER


class OllamaStreamingSafety:
    def __init__(self, span, llm_request_type, span_name: str):
        self._llm_request_type = llm_request_type
        self._streams = CompletionTextStreamGroup(
            span=span,
            provider=PROVIDER,
            span_name=span_name,
            request_type=llm_request_type.value,
        )

    def process_chunk(self, chunk):
        key = self._stream_key()
        text = self._chunk_text(chunk)
        if isinstance(text, str):
            masked = self._streams.process(
                key,
                text,
                segment_index=0,
                segment_role="assistant",
            )
            self._set_chunk_text(chunk, masked)

        if get_object_value(chunk, "done"):
            tail = self._streams.flush(key)
            if tail:
                current_text = self._chunk_text(chunk) or ""
                self._set_chunk_text(chunk, f"{current_text}{tail}")

        return chunk

    def _chunk_text(self, chunk):
        if self._llm_request_type.value == "chat":
            message = get_object_value(chunk, "message")
            return get_object_value(message, "content")
        return get_object_value(chunk, "response")

    def _set_chunk_text(self, chunk, text: str) -> None:
        if self._llm_request_type.value == "chat":
            message = get_object_value(chunk, "message")
            if message is not None:
                set_object_value(message, "content", text)
            return
        set_object_value(chunk, "response", text)

    def _stream_key(self):
        return "chat" if self._llm_request_type.value == "chat" else "completion"
