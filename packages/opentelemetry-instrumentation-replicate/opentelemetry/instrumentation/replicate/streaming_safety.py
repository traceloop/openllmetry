from __future__ import annotations

from collections.abc import Iterator

from opentelemetry.instrumentation.fortifyroot.text_streaming import (
    CompletionTextStreamGroup,
)
from opentelemetry.instrumentation.replicate.safety import PROVIDER
from opentelemetry.semconv_ai import LLMRequestTypeValues


class ReplicateStreamingSafety:
    def __init__(self, span, span_name: str):
        self._streams = CompletionTextStreamGroup(
            span=span,
            provider=PROVIDER,
            span_name=span_name,
            request_type=LLMRequestTypeValues.COMPLETION.value,
        )

    def process_text(self, text: str) -> str:
        return self._streams.process(
            "completion",
            text,
            segment_index=0,
            segment_role="assistant",
        )

    def flush(self) -> str:
        return self._streams.flush("completion")


def build_streaming_response(
    response,
    *,
    span,
    finalize_response,
) -> Iterator[str]:
    complete_response = ""
    streaming_safety = ReplicateStreamingSafety(span, "replicate.stream")
    pending_item = None
    for item in response:
        masked_item = streaming_safety.process_text(str(item))
        if pending_item is not None:
            complete_response += pending_item
            yield pending_item
        pending_item = masked_item

    if pending_item is not None:
        tail = streaming_safety.flush()
        pending_item = f"{pending_item}{tail}" if tail else pending_item
        complete_response += pending_item
        yield pending_item

    finalize_response(complete_response)
