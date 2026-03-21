from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

from opentelemetry.instrumentation.fortifyroot import get_object_value, set_object_value
from opentelemetry.instrumentation.fortifyroot.text_streaming import (
    CompletionTextStreamGroup,
)
from opentelemetry.instrumentation.vertexai.safety import PROVIDER
from opentelemetry.semconv_ai import LLMRequestTypeValues


class VertexAIStreamingSafety:
    def __init__(self, span, span_name: str):
        self._streams = CompletionTextStreamGroup(
            span=span,
            provider=PROVIDER,
            span_name=span_name,
            request_type=LLMRequestTypeValues.COMPLETION.value,
        )

    def process_item(self, item):
        parts = self._text_parts(item)
        for key, part in parts:
            text = get_object_value(part, "text")
            if not isinstance(text, str):
                continue
            masked = self._streams.process(
                key,
                text,
                segment_index=key[0],
                segment_role="assistant",
                metadata={"part_index": key[1]},
            )
            set_object_value(part, "text", masked)
        self._sync_item_text(item)
        return item

    def flush_pending_item(self, item):
        parts = self._text_parts(item)
        if not parts:
            return
        for key, part in parts:
            tail = self._streams.flush(key)
            if not tail:
                continue
            current_text = get_object_value(part, "text") or ""
            set_object_value(part, "text", f"{current_text}{tail}")
        self._sync_item_text(item)

    def _text_parts(self, item):
        parts = []
        for candidate_index, candidate in enumerate(get_object_value(item, "candidates") or []):
            content = get_object_value(candidate, "content")
            for part_index, part in enumerate(get_object_value(content, "parts") or []):
                if isinstance(get_object_value(part, "text"), str):
                    parts.append(((candidate_index, part_index), part))
        return parts

    def _sync_item_text(self, item):
        parts = self._text_parts(item)
        if not parts:
            return
        text_parts = [get_object_value(part, "text") for _, part in parts]
        combined = "".join(part for part in text_parts if isinstance(part, str))
        set_object_value(item, "text", combined)


def build_streaming_response(
    response,
    *,
    span,
    llm_model,
    finalize_response,
    span_name=None,
) -> Iterator:
    complete_response = ""
    token_usage = None
    pending_item = None
    streaming_safety = VertexAIStreamingSafety(
        span,
        span_name or "vertexai.generate_content",
    )
    for item in response:
        item = streaming_safety.process_item(item)
        if pending_item is not None:
            yield pending_item
            complete_response += str(pending_item.text)
            if pending_item.usage_metadata:
                token_usage = pending_item.usage_metadata
        pending_item = item

    if pending_item is not None:
        streaming_safety.flush_pending_item(pending_item)
        yield pending_item
        complete_response += str(pending_item.text)
        if pending_item.usage_metadata:
            token_usage = pending_item.usage_metadata

    finalize_response(complete_response, token_usage, llm_model)


async def build_async_streaming_response(
    response,
    *,
    span,
    llm_model,
    finalize_response,
    span_name=None,
) -> AsyncIterator:
    complete_response = ""
    token_usage = None
    pending_item = None
    streaming_safety = VertexAIStreamingSafety(
        span,
        span_name or "vertexai.generate_content",
    )
    async for item in response:
        item = streaming_safety.process_item(item)
        if pending_item is not None:
            yield pending_item
            complete_response += str(pending_item.text)
            if pending_item.usage_metadata:
                token_usage = pending_item.usage_metadata
        pending_item = item

    if pending_item is not None:
        streaming_safety.flush_pending_item(pending_item)
        yield pending_item
        complete_response += str(pending_item.text)
        if pending_item.usage_metadata:
            token_usage = pending_item.usage_metadata

    finalize_response(complete_response, token_usage, llm_model)
