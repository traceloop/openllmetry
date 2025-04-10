import json
from typing import Union

from opentelemetry._events import Event, EventLogger
from opentelemetry.instrumentation.bedrock.utils import (
    dont_throw,
    get_event_attributes,
    is_content_enabled,
)
from wrapt import ObjectProxy


class StreamingWrapper(ObjectProxy):
    def __init__(
        self,
        response,
        stream_done_callback=None,
        emit_events_otel: bool = False,
        event_logger: Union[EventLogger, None] = None,
    ):
        super().__init__(response)

        self._stream_done_callback = stream_done_callback
        self._accumulating_body = {}
        self._emit_events_otel = emit_events_otel
        self._event_logger = event_logger

    def __iter__(self):
        it = iter(self.__wrapped__)
        done = False
        while not done:
            try:
                event = next(it)
                self._process_event(event)
                yield event
            except StopIteration:
                done = True
                if self._emit_events_otel and self._event_logger is not None:
                    self._emit_response_event()
                if self._stream_done_callback:
                    self._stream_done_callback(self._accumulating_body)

    @dont_throw
    def _process_event(self, event):
        chunk = event.get("chunk")
        if not chunk:
            return

        decoded_chunk = json.loads(chunk.get("bytes").decode())
        type = decoded_chunk.get("type")
        if type is None:
            self._accumulate_events(decoded_chunk)
        elif type == "message_start":
            self._accumulating_body = decoded_chunk.get("message")
        elif type == "content_block_start":
            self._accumulating_body["content"].append(
                decoded_chunk.get("content_block")
            )
        elif type == "content_block_delta":
            self._accumulating_body["content"][-1]["text"] += decoded_chunk.get(
                "delta"
            ).get("text")
        elif type == "message_stop":
            self._accumulating_body["invocation_metrics"] = decoded_chunk.get(
                "amazon-bedrock-invocationMetrics"
            )

    def _accumulate_events(self, event):
        print(self._accumulating_body)
        for key in event:
            if key == "contentBlockDelta":
                delta = event.get(key).get("delta", {}).get("text")
                if "outputText" in self._accumulating_body:
                    self._accumulating_body["outputText"] += delta
                else:
                    self._accumulating_body["outputText"] = delta
            elif key in self._accumulating_body:
                self._accumulating_body[key] += event.get(key)
            elif key == "messageStop":
                self._accumulating_body["stop_reason"] = event.get(key).get(
                    "stopReason"
                )
            else:
                self._accumulating_body[key] = event.get(key)

    def _emit_response_event(self):
        attributes = get_event_attributes()
        body = {
            "index": 0,
            "finish_reason": self._accumulating_body.get("stop_reason") or "unknown",
            "message": {},
        }

        if is_content_enabled():
            body["message"]["content"] = self._accumulating_body.get(
                "content"
            ) or self._accumulating_body.get("outputText")

        self._event_logger.emit(
            Event(name="gen_ai.choice", body=body, attributes=attributes)
        )
