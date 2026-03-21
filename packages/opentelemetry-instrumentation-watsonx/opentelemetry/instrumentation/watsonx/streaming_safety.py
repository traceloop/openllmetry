from __future__ import annotations

from collections.abc import Iterator

from opentelemetry.instrumentation.fortifyroot.text_streaming import (
    CompletionTextStreamGroup,
)
from opentelemetry.instrumentation.watsonx.safety import PROVIDER
from opentelemetry.semconv_ai import LLMRequestTypeValues


class WatsonxStreamingSafety:
    def __init__(self, span, span_name: str):
        self._streams = CompletionTextStreamGroup(
            span=span,
            provider=PROVIDER,
            span_name=span_name,
            request_type=LLMRequestTypeValues.COMPLETION.value,
        )

    def process_item(self, item):
        try:
            text = item["results"][0]["generated_text"]
            item["results"][0]["generated_text"] = self._streams.process(
                "completion",
                text,
                segment_index=0,
                segment_role="assistant",
            )
        except Exception:
            return item
        return item

    def flush_pending_item(self, item):
        tail = self._streams.flush("completion")
        if not tail:
            return
        item["results"][0]["generated_text"] += tail


def build_streaming_response(
    response,
    *,
    span,
    raw_flag,
    finalize_response,
    span_name=None,
) -> Iterator:
    stream_state = {
        "generated_text": "",
        "model_id": "",
        "stop_reason": "",
        "generated_token_count": 0,
        "input_token_count": 0,
    }
    pending_item = None
    streaming_safety = WatsonxStreamingSafety(
        span,
        span_name or "watsonx.generate_text_stream",
    )
    for item in response:
        try:
            item = streaming_safety.process_item(item)
        except Exception:
            pass
        if pending_item is not None:
            _update_stream_state(stream_state, pending_item)
            yield pending_item if raw_flag else pending_item["results"][0]["generated_text"]
        pending_item = item

    if pending_item is not None:
        streaming_safety.flush_pending_item(pending_item)
        _update_stream_state(stream_state, pending_item)
        yield pending_item if raw_flag else pending_item["results"][0]["generated_text"]

    finalize_response(stream_state)


def _update_stream_state(stream_state, item):
    stream_state["model_id"] = item["model_id"]
    stream_state["generated_text"] += item["results"][0]["generated_text"]
    stream_state["input_token_count"] += item["results"][0]["input_token_count"]
    stream_state["generated_token_count"] = item["results"][0]["generated_token_count"]
    stream_state["stop_reason"] = item["results"][0]["stop_reason"]
