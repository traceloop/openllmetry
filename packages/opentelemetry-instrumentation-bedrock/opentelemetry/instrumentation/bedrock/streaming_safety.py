from __future__ import annotations

import json

from wrapt import ObjectProxy

from opentelemetry.instrumentation.bedrock.event_emitter import (
    emit_streaming_converse_response_event,
)
from opentelemetry.instrumentation.bedrock.guardrail import guardrail_converse
from opentelemetry.instrumentation.bedrock.safety import PROVIDER, _request_type
from opentelemetry.instrumentation.bedrock.span_utils import (
    converse_usage_record,
    set_converse_streaming_response_span_attributes,
)
from opentelemetry.instrumentation.bedrock.utils import should_emit_events
from opentelemetry.instrumentation.fortifyroot.text_streaming import (
    CompletionTextStreamGroup,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues


class _BedrockChunkStreamingSafety:
    def __init__(self, *, span, span_name: str, request_type: str):
        self._streams = CompletionTextStreamGroup(
            span=span,
            provider=PROVIDER,
            span_name=span_name,
            request_type=request_type,
        )

    def process_event(self, event):
        payload = _decode_chunk_event(event)
        key = self._payload_key(payload)
        text = self._payload_text(payload)
        if key is None or text is None:
            return event

        masked = self._streams.process(
            key,
            text,
            segment_index=key[1] if isinstance(key, tuple) and len(key) > 1 else 0,
            segment_role="assistant",
        )
        _set_payload_text(payload, masked)
        _encode_chunk_event(event, payload)
        return event

    def flush_transition(self, pending_event, current_event):
        pending_payload = _decode_chunk_event(pending_event)
        pending_key = self._payload_key(pending_payload)
        if pending_key is None:
            return

        current_payload = _decode_chunk_event(current_event)
        current_key = self._payload_key(current_payload)
        if current_key == pending_key:
            return

        self._append_tail(pending_event, pending_key)

    def flush_pending_event(self, event):
        payload = _decode_chunk_event(event)
        key = self._payload_key(payload)
        if key is None:
            return
        self._append_tail(event, key)

    def _append_tail(self, event, key):
        tail = self._streams.flush(key)
        if not tail:
            return

        payload = _decode_chunk_event(event)
        if self._payload_key(payload) != key:
            return

        current_text = self._payload_text(payload) or ""
        _set_payload_text(payload, f"{current_text}{tail}")
        _encode_chunk_event(event, payload)

    def _payload_key(self, payload):
        if not isinstance(payload, dict):
            return None

        if payload.get("type") == "content_block_delta":
            delta = payload.get("delta") or {}
            if isinstance(delta.get("text"), str):
                return ("content", payload.get("index", 0) or 0)

        if payload.get("type") == "content_block_start":
            block = payload.get("content_block") or {}
            if isinstance(block.get("text"), str):
                return ("content", payload.get("index", 0) or 0)

        if "contentBlockDelta" in payload:
            delta = (payload.get("contentBlockDelta") or {}).get("delta") or {}
            if isinstance(delta.get("text"), str):
                return (
                    "content",
                    (payload.get("contentBlockDelta") or {}).get("contentBlockIndex", 0)
                    or payload.get("contentBlockIndex", 0)
                    or 0,
                )

        if "contentBlockStart" in payload:
            block = (payload.get("contentBlockStart") or {}).get("start") or {}
            if isinstance(block.get("text"), str):
                return (
                    "content",
                    (payload.get("contentBlockStart") or {}).get("contentBlockIndex", 0)
                    or payload.get("contentBlockIndex", 0)
                    or 0,
                )

        for key in ("completion", "outputText", "generated_text", "text"):
            if isinstance(payload.get(key), str):
                return ("choice", 0)

        return None

    def _payload_text(self, payload):
        if not isinstance(payload, dict):
            return None

        if payload.get("type") == "content_block_delta":
            return ((payload.get("delta") or {}).get("text"))

        if payload.get("type") == "content_block_start":
            return ((payload.get("content_block") or {}).get("text"))

        if "contentBlockDelta" in payload:
            return (((payload.get("contentBlockDelta") or {}).get("delta") or {}).get("text"))

        if "contentBlockStart" in payload:
            return (((payload.get("contentBlockStart") or {}).get("start") or {}).get("text"))

        for key in ("completion", "outputText", "generated_text", "text"):
            text = payload.get(key)
            if isinstance(text, str):
                return text

        return None


class BedrockInvokeSafetyStreamingWrapper(ObjectProxy):
    def __init__(self, response, *, span, stream_done_callback=None):
        super().__init__(response)

        self._self_stream_done_callback = stream_done_callback
        self._self_accumulating_body = {}
        self._self_pending_event = None
        self._self_streaming_safety = _BedrockChunkStreamingSafety(
            span=span,
            span_name=getattr(span, "name", "bedrock.completion"),
            request_type=_request_type(getattr(span, "name", "bedrock.completion")),
        )

    def __iter__(self):
        for event in self.__wrapped__:
            event = self._self_streaming_safety.process_event(event)
            if self._self_pending_event is None:
                self._self_pending_event = event
                continue

            self._self_streaming_safety.flush_transition(self._self_pending_event, event)
            self._accumulate_event(self._self_pending_event)
            yield self._self_pending_event
            self._self_pending_event = event

        if self._self_pending_event is not None:
            self._self_streaming_safety.flush_pending_event(self._self_pending_event)
            self._accumulate_event(self._self_pending_event)
            yield self._self_pending_event

        if self._self_stream_done_callback:
            self._self_stream_done_callback(self._self_accumulating_body)

    def _accumulate_event(self, event):
        payload = _decode_chunk_event(event)
        if not isinstance(payload, dict):
            return

        event_type = payload.get("type")
        if event_type is None:
            self._accumulate_events(payload)
        elif event_type == "message_start":
            self._self_accumulating_body = payload.get("message") or {}
        elif event_type == "content_block_start":
            content = self._self_accumulating_body.setdefault("content", [])
            content.append(payload.get("content_block"))
        elif event_type == "content_block_delta":
            content = self._self_accumulating_body.setdefault("content", [])
            if not content:
                content.append({"text": ""})
            delta_text = (payload.get("delta") or {}).get("text")
            if isinstance(delta_text, str):
                content[-1]["text"] = f"{content[-1].get('text', '')}{delta_text}"
        elif event_type == "message_stop":
            self._self_accumulating_body["invocation_metrics"] = payload.get(
                "amazon-bedrock-invocationMetrics"
            )

    def _accumulate_events(self, payload):
        for key, value in payload.items():
            if key == "contentBlockDelta":
                delta = (value or {}).get("delta", {}).get("text")
                if isinstance(delta, str):
                    if "outputText" in self._self_accumulating_body:
                        self._self_accumulating_body["outputText"] += delta
                    else:
                        self._self_accumulating_body["outputText"] = delta
            elif key == "messageStop":
                self._self_accumulating_body["stop_reason"] = (value or {}).get("stopReason")
            elif key in self._self_accumulating_body and isinstance(value, str):
                self._self_accumulating_body[key] += value
            else:
                self._self_accumulating_body[key] = value


class _BedrockConverseStreamingSafety:
    def __init__(self, *, span, span_name: str):
        self._streams = CompletionTextStreamGroup(
            span=span,
            provider=PROVIDER,
            span_name=span_name,
            request_type=LLMRequestTypeValues.CHAT.value,
        )

    def process_event(self, event):
        key = self._event_key(event)
        text = self._event_text(event)
        if key is None or text is None:
            return event

        masked = self._streams.process(
            key,
            text,
            segment_index=key[1] if isinstance(key, tuple) and len(key) > 1 else 0,
            segment_role="assistant",
        )
        self._set_event_text(event, masked)
        return event

    def flush_transition(self, pending_event, current_event):
        pending_key = self._event_key(pending_event)
        if pending_key is None:
            return

        current_key = self._event_key(current_event)
        if current_key == pending_key:
            return

        self._append_tail(pending_event, pending_key)

    def flush_pending_event(self, event):
        key = self._event_key(event)
        if key is None:
            return
        self._append_tail(event, key)

    def _append_tail(self, event, key):
        tail = self._streams.flush(key)
        if not tail or self._event_key(event) != key:
            return
        current_text = self._event_text(event) or ""
        self._set_event_text(event, f"{current_text}{tail}")

    def _event_key(self, event):
        if not isinstance(event, dict):
            return None

        if "contentBlockDelta" in event:
            delta = (event.get("contentBlockDelta") or {}).get("delta") or {}
            if isinstance(delta.get("text"), str):
                return (
                    "content",
                    (event.get("contentBlockDelta") or {}).get("contentBlockIndex", 0)
                    or event.get("contentBlockIndex", 0)
                    or 0,
                )

        if "contentBlockStart" in event:
            start = (event.get("contentBlockStart") or {}).get("start") or {}
            if isinstance(start.get("text"), str):
                return (
                    "content",
                    (event.get("contentBlockStart") or {}).get("contentBlockIndex", 0)
                    or event.get("contentBlockIndex", 0)
                    or 0,
                )

        return None

    def _event_text(self, event):
        if not isinstance(event, dict):
            return None

        if "contentBlockDelta" in event:
            return (((event.get("contentBlockDelta") or {}).get("delta") or {}).get("text"))

        if "contentBlockStart" in event:
            return (((event.get("contentBlockStart") or {}).get("start") or {}).get("text"))

        return None

    def _set_event_text(self, event, text):
        if "contentBlockDelta" in event:
            event["contentBlockDelta"]["delta"]["text"] = text
        elif "contentBlockStart" in event:
            event["contentBlockStart"]["start"]["text"] = text


class BedrockConverseSafetyStream(ObjectProxy):
    def __init__(
        self,
        response,
        *,
        span,
        provider,
        model,
        metric_params,
        event_logger,
    ):
        super().__init__(response)

        self._self_span = span
        self._self_provider = provider
        self._self_model = model
        self._self_metric_params = metric_params
        self._self_event_logger = event_logger
        self._self_pending_event = None
        self._self_role = "unknown"
        self._self_response_msg = []
        self._self_span_ended = False
        self._self_streaming_safety = _BedrockConverseStreamingSafety(
            span=span,
            span_name=getattr(span, "name", "bedrock.converse"),
        )

    def __iter__(self):
        for event in self.__wrapped__:
            event = self._self_streaming_safety.process_event(event)
            if self._self_pending_event is None:
                self._self_pending_event = event
                continue

            self._self_streaming_safety.flush_transition(self._self_pending_event, event)
            self._observe_event(self._self_pending_event)
            yield self._self_pending_event
            self._self_pending_event = event

        if self._self_pending_event is not None:
            self._self_streaming_safety.flush_pending_event(self._self_pending_event)
            self._observe_event(self._self_pending_event)
            yield self._self_pending_event

        if not self._self_span_ended:
            self._self_span.end()
            self._self_span_ended = True

    def _observe_event(self, event):
        if not isinstance(event, dict):
            return

        if "messageStart" in event:
            self._self_role = (event.get("messageStart") or {}).get("role", self._self_role)

        if "contentBlockDelta" in event:
            delta_text = ((event.get("contentBlockDelta") or {}).get("delta") or {}).get("text")
            if isinstance(delta_text, str):
                self._self_response_msg.append(delta_text)
        elif "contentBlockStart" in event:
            start_text = ((event.get("contentBlockStart") or {}).get("start") or {}).get("text")
            if isinstance(start_text, str):
                self._self_response_msg.append(start_text)

        if "messageStop" in event:
            stop_reason = (event.get("messageStop") or {}).get("stopReason", "unknown")
            if should_emit_events() and self._self_event_logger:
                emit_streaming_converse_response_event(
                    self._self_event_logger,
                    self._self_response_msg,
                    self._self_role,
                    stop_reason,
                )
            else:
                set_converse_streaming_response_span_attributes(
                    self._self_response_msg,
                    self._self_role,
                    self._self_span,
                )

        if "metadata" in event:
            metadata = event.get("metadata")
            guardrail_converse(
                self._self_span,
                metadata,
                self._self_provider,
                self._self_model,
                self._self_metric_params,
            )
            converse_usage_record(self._self_span, metadata, self._self_metric_params)
            if not self._self_span_ended:
                self._self_span.end()
                self._self_span_ended = True


def create_invoke_stream_wrapper(response, *, span, stream_done_callback=None):
    return BedrockInvokeSafetyStreamingWrapper(
        response,
        span=span,
        stream_done_callback=stream_done_callback,
    )


def create_converse_stream_wrapper(
    response,
    *,
    span,
    provider,
    model,
    metric_params,
    event_logger,
):
    return BedrockConverseSafetyStream(
        response,
        span=span,
        provider=provider,
        model=model,
        metric_params=metric_params,
        event_logger=event_logger,
    )


def _decode_chunk_event(event):
    chunk = event.get("chunk") if isinstance(event, dict) else None
    if not isinstance(chunk, dict):
        return None

    raw_bytes = chunk.get("bytes")
    if not isinstance(raw_bytes, (bytes, bytearray)):
        return None

    try:
        return json.loads(raw_bytes.decode("utf-8"))
    except Exception:
        return None


def _encode_chunk_event(event, payload):
    if not isinstance(event, dict):
        return
    chunk = event.get("chunk")
    if not isinstance(chunk, dict):
        return
    chunk["bytes"] = json.dumps(payload).encode("utf-8")


def _set_payload_text(payload, text):
    if payload.get("type") == "content_block_delta":
        payload["delta"]["text"] = text
        return

    if payload.get("type") == "content_block_start":
        payload["content_block"]["text"] = text
        return

    if "contentBlockDelta" in payload:
        payload["contentBlockDelta"]["delta"]["text"] = text
        return

    if "contentBlockStart" in payload:
        payload["contentBlockStart"]["start"]["text"] = text
        return

    for key in ("completion", "outputText", "generated_text", "text"):
        if isinstance(payload.get(key), str):
            payload[key] = text
            return
