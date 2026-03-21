from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

from opentelemetry.trace import Span

from opentelemetry.instrumentation.fortifyroot.safety import SafetyLocation
from opentelemetry.instrumentation.fortifyroot.streaming import (
    BoundCompletionSafetyStream,
    create_completion_safety_stream,
)


class CompletionTextStreamGroup:
    """Track per-segment completion safety streams for a single response."""

    def __init__(
        self,
        *,
        span: Span | None,
        provider: str,
        span_name: str,
        request_type: str | None,
    ) -> None:
        self._span = span
        self._provider = provider
        self._span_name = span_name
        self._request_type = request_type
        self._streams: dict[Hashable, BoundCompletionSafetyStream | None] = {}

    def process(
        self,
        key: Hashable,
        text: str | None,
        *,
        segment_index: int | None,
        segment_role: str | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Process a streaming chunk for the given segment key."""
        if text is None or text == "":
            return text or ""

        stream = self._get_stream(
            key,
            segment_index=segment_index,
            segment_role=segment_role,
            metadata=metadata,
        )
        if stream is None:
            return text
        return stream.process_chunk(text)

    def flush(self, key: Hashable) -> str:
        """Flush the trailing buffered text for a single segment."""

        stream = self._streams.get(key)
        if stream is None:
            return ""
        return stream.flush()

    def flush_all(self) -> dict[Hashable, str]:
        """Flush all tracked segments and return non-empty tails."""

        flushed: dict[Hashable, str] = {}
        for key, stream in self._streams.items():
            if stream is None:
                continue
            text = stream.flush()
            if text:
                flushed[key] = text
        return flushed

    def _get_stream(
        self,
        key: Hashable,
        *,
        segment_index: int | None,
        segment_role: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> BoundCompletionSafetyStream | None:
        if key not in self._streams:
            self._streams[key] = create_completion_safety_stream(
                span=self._span,
                provider=self._provider,
                span_name=self._span_name,
                location=SafetyLocation.COMPLETION,
                request_type=self._request_type,
                segment_index=segment_index,
                segment_role=segment_role,
                metadata=metadata,
            )
        return self._streams[key]
