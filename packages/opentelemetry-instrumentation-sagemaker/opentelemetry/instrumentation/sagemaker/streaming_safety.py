from __future__ import annotations

from wrapt import ObjectProxy

from opentelemetry.instrumentation.fortifyroot.text_streaming import (
    CompletionTextStreamGroup,
)
from opentelemetry.instrumentation.sagemaker.safety import PROVIDER
from opentelemetry.semconv_ai import LLMRequestTypeValues


class SageMakerSafetyStreamingWrapper(ObjectProxy):
    def __init__(self, response, *, span, stream_done_callback=None):
        super().__init__(response)

        self._self_stream_done_callback = stream_done_callback
        self._self_accumulating_body = ""
        self._self_pending_event = None
        self._self_streams = CompletionTextStreamGroup(
            span=span,
            provider=PROVIDER,
            span_name=getattr(span, "name", "sagemaker.completion"),
            request_type=LLMRequestTypeValues.COMPLETION.value,
        )

    def __iter__(self):
        for event in self.__wrapped__:
            event = self._process_event(event)
            if self._self_pending_event is None:
                self._self_pending_event = event
                continue

            self._accumulate_event(self._self_pending_event)
            yield self._self_pending_event
            self._self_pending_event = event

        if self._self_pending_event is not None:
            self._flush_pending_event(self._self_pending_event)
            self._accumulate_event(self._self_pending_event)
            yield self._self_pending_event

        if self._self_stream_done_callback:
            self._self_stream_done_callback(self._self_accumulating_body)

    def _process_event(self, event):
        payload_part = event.get("PayloadPart") if isinstance(event, dict) else None
        if not isinstance(payload_part, dict):
            return event

        raw_bytes = payload_part.get("Bytes")
        if not isinstance(raw_bytes, (bytes, bytearray)):
            return event

        try:
            decoded_text = raw_bytes.decode("utf-8")
        except Exception:
            return event

        masked_text = self._self_streams.process(
            ("choice", 0),
            decoded_text,
            segment_index=0,
            segment_role="assistant",
        )
        payload_part["Bytes"] = masked_text.encode("utf-8")
        return event

    def _flush_pending_event(self, event):
        payload_part = event.get("PayloadPart") if isinstance(event, dict) else None
        if not isinstance(payload_part, dict):
            return

        tail = self._self_streams.flush(("choice", 0))
        if not tail:
            return

        raw_bytes = payload_part.get("Bytes")
        if not isinstance(raw_bytes, (bytes, bytearray)):
            return

        try:
            current_text = raw_bytes.decode("utf-8")
        except Exception:
            return

        payload_part["Bytes"] = f"{current_text}{tail}".encode("utf-8")

    def _accumulate_event(self, event):
        payload_part = event.get("PayloadPart") if isinstance(event, dict) else None
        if not isinstance(payload_part, dict):
            return

        raw_bytes = payload_part.get("Bytes")
        if not isinstance(raw_bytes, (bytes, bytearray)):
            return

        try:
            decoded_text = raw_bytes.decode("utf-8")
        except Exception:
            return

        self._self_accumulating_body += decoded_text


def create_streaming_wrapper(response, *, span, stream_done_callback=None):
    return SageMakerSafetyStreamingWrapper(
        response,
        span=span,
        stream_done_callback=stream_done_callback,
    )
