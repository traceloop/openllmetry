import json
import logging

from opentelemetry.instrumentation.bedrock.utils import (
    dont_throw,
)
from wrapt import ObjectProxy

logger = logging.getLogger(__name__)


def _accumulate_openai_chunk(body, chunk):
    """Fold an OpenAI chat.completion.chunk (openai.* models) into a
    chat.completion-shaped body, which the non-streaming span helpers read."""
    body["id"] = chunk.get("id")
    body["model"] = chunk.get("model")
    if chunk.get("usage"):
        body["usage"] = chunk["usage"]
    if chunk.get("amazon-bedrock-invocationMetrics"):
        body["invocation_metrics"] = chunk["amazon-bedrock-invocationMetrics"]
    for key in ("amazon-bedrock-guardrailAction", "amazon-bedrock-trace"):  # read by guardrail_handling
        if key in chunk:
            body[key] = chunk[key]
    choices = body.setdefault("choices", [])
    for choice in chunk.get("choices") or []:
        index = choice.get("index", 0)
        while len(choices) <= index:
            choices.append({"message": {"role": "assistant", "content": ""}})
        message = choices[index]["message"]
        delta = choice.get("delta") or {}
        message["content"] += delta.get("content") or ""
        for tool_call in delta.get("tool_calls") or []:
            tool_calls = message.setdefault("tool_calls", [])
            function = tool_call.get("function") or {}
            if tool_call.get("id"):
                tool_calls.append({
                    "id": tool_call["id"],
                    "function": {"name": function.get("name"), "arguments": ""},
                })
            position = tool_call.get("index", len(tool_calls) - 1)
            if 0 <= position < len(tool_calls):
                tool_calls[position]["function"]["arguments"] += function.get("arguments") or ""
        if choice.get("finish_reason"):
            choices[index]["finish_reason"] = choice["finish_reason"]


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
        if decoded_chunk.get("object") == "chat.completion.chunk":
            _accumulate_openai_chunk(self._accumulating_body, decoded_chunk)
        elif type is None:
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
                if self._stream_done_callback:
                    self._stream_done_callback(self._accumulating_body)

    @dont_throw
    def _process_event(self, event):
        chunk = event.get("chunk")
        if not chunk:
            return

        decoded_chunk = json.loads(chunk.get("bytes").decode())
        type = decoded_chunk.get("type")
        if decoded_chunk.get("object") == "chat.completion.chunk":
            _accumulate_openai_chunk(self._accumulating_body, decoded_chunk)
        elif type is None:
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
