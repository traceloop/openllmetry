import json
import logging

from opentelemetry.instrumentation.bedrock.utils import (
    dont_throw,
)
from wrapt import ObjectProxy

logger = logging.getLogger(__name__)


class AsyncStreamingWrapper(ObjectProxy):
    """Async counterpart of StreamingWrapper for aioboto3's EventStream."""

    def __init__(self, response, stream_done_callback=None):
        super().__init__(response)
        self._stream_done_callback = stream_done_callback
        self._accumulating_body = {}
        self._done = False

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        # try/finally ensures the callback (which ends the span) always fires,
        # even if the caller breaks out of `async for` early or the underlying
        # stream raises mid-iteration. The `_done` guard prevents double-firing
        # if the wrapper is iterated more than once.
        try:
            async for event in self.__wrapped__:
                self._process_event(event)
                yield event
        finally:
            if self._stream_done_callback and not self._done:
                self._done = True
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
            delta = decoded_chunk.get("delta", {})
            if delta.get("text") is not None:
                self._accumulating_body["content"][-1]["text"] += delta["text"]
            elif delta.get("type") == "input_json_delta":
                partial_json = delta.get("partial_json", "")
                current = self._accumulating_body["content"][-1]
                current.setdefault("input", "")
                current["input"] += partial_json
            elif delta.get("type") == "thinking_delta":
                thinking_text = delta.get("thinking", "")
                current = self._accumulating_body["content"][-1]
                current.setdefault("thinking", "")
                current["thinking"] += thinking_text
        elif type == "message_delta":
            delta = decoded_chunk.get("delta", {})
            if delta.get("stop_reason"):
                self._accumulating_body["stop_reason"] = delta["stop_reason"]
            if decoded_chunk.get("usage"):
                usage = self._accumulating_body.get("usage", {})
                usage.update(decoded_chunk["usage"])
                self._accumulating_body["usage"] = usage
        elif type == "message_stop":
            self._accumulating_body["invocation_metrics"] = decoded_chunk.get(
                "amazon-bedrock-invocationMetrics"
            )

    def _accumulate_events(self, event):
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


class StreamingWrapper(ObjectProxy):
    def __init__(
        self,
        response,
        stream_done_callback=None,
    ):
        super().__init__(response)

        self._stream_done_callback = stream_done_callback
        self._accumulating_body = {}
        self._done = False

    def __iter__(self):
        it = iter(self.__wrapped__)
        # try/finally ensures the callback (which ends the span) always fires,
        # even if the caller breaks out of `for` early, closes the stream, or
        # drops the iterator. The `_done` guard prevents double-firing if the
        # wrapper is iterated more than once. Mirrors AsyncStreamingWrapper.
        try:
            while True:
                try:
                    event = next(it)
                except StopIteration:
                    break
                self._process_event(event)
                yield event
        finally:
            self._ensure_done()

    def close(self):
        # End the span promptly when the caller closes the stream instead of
        # waiting for the abandoned iterator to be garbage-collected, then
        # delegate to the wrapped stream's own close().
        self._ensure_done()
        close = getattr(self.__wrapped__, "close", None)
        if callable(close):
            close()

    def _ensure_done(self):
        if self._stream_done_callback and not self._done:
            self._done = True
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
            delta = decoded_chunk.get("delta", {})
            if delta.get("text") is not None:
                self._accumulating_body["content"][-1]["text"] += delta["text"]
            elif delta.get("type") == "input_json_delta":
                partial_json = delta.get("partial_json", "")
                current = self._accumulating_body["content"][-1]
                current.setdefault("input", "")
                current["input"] += partial_json
            elif delta.get("type") == "thinking_delta":
                thinking_text = delta.get("thinking", "")
                current = self._accumulating_body["content"][-1]
                current.setdefault("thinking", "")
                current["thinking"] += thinking_text
        elif type == "message_delta":
            delta = decoded_chunk.get("delta", {})
            if delta.get("stop_reason"):
                self._accumulating_body["stop_reason"] = delta["stop_reason"]
            if decoded_chunk.get("usage"):
                usage = self._accumulating_body.get("usage", {})
                usage.update(decoded_chunk["usage"])
                self._accumulating_body["usage"] = usage
        elif type == "message_stop":
            self._accumulating_body["invocation_metrics"] = decoded_chunk.get(
                "amazon-bedrock-invocationMetrics"
            )

    def _accumulate_events(self, event):
        logger.debug("Accumulating body: %s", self._accumulating_body)
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
