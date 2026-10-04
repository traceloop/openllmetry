"""Streaming (SSE) support for OCI Generative AI ``chat`` / ``generate_text`` calls.

With ``is_stream=True`` the OCI SDK returns an ``oci.response.Response`` whose ``data`` is an
``SSEClient``; callers iterate ``response.data.events()`` and read ``event.data`` (a JSON document or the
``[DONE]`` sentinel). Observed event payloads:

GENERIC api format::

    {"index": 0, "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "Hello"}]}, "pad": "aaa"}
    {"message": {...}, "finishReason": "stop", "pad": "a"}
    {"usage": {"completionTokens": 8, "promptTokens": 41, "totalTokens": 49}, "pad": "aa"}   # is_include_usage
    [DONE]

COHERE api format::

    {"apiFormat": "COHERE", "text": "Hi", "pad": "aaa"}
    {"apiFormat": "COHERE", "text": "<full text>", "chatHistory": [...], "finishReason": "COMPLETE",
     "usage": {...}}
"""

import json
import logging
from contextlib import nullcontext

from opentelemetry.trace import use_span
from wrapt import ObjectProxy

logger = logging.getLogger(__name__)

DONE_SENTINEL = "[DONE]"


class StreamAccumulator:
    """Aggregates SSE event payloads into per-choice text, tool calls, finish reasons and usage."""

    def __init__(self):
        self.api_format = None
        self.usage = None
        self.response_id = None
        self._choices = {}

    def _choice(self, index):
        return self._choices.setdefault(index, {"text": "", "reasoning": "", "tool_calls": {}, "finish_reason": None})

    def process(self, raw):
        if raw is None:
            return
        raw = raw.strip() if isinstance(raw, str) else raw
        if not raw or raw == DONE_SENTINEL:
            return
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            logger.debug("Skipping non-JSON OCI GenAI stream event: %r", raw)
            return
        if not isinstance(payload, dict):
            return

        if payload.get("apiFormat"):
            self.api_format = payload["apiFormat"]
        if isinstance(payload.get("usage"), dict):
            self.usage = payload["usage"]
        if payload.get("id"):
            self.response_id = payload["id"]

        message = payload.get("message")
        if isinstance(message, dict):
            self._process_message(payload, message)
        elif "text" in payload:
            self._process_text(payload)
        elif payload.get("finishReason"):
            self._choice(payload.get("index", 0))["finish_reason"] = payload["finishReason"]

    def _process_message(self, payload, message):
        """GENERIC / COHEREV2 shaped events: partial ``message`` objects, one per choice index."""
        choice = self._choice(payload.get("index", 0))
        content = message.get("content")
        if isinstance(content, str):
            choice["text"] += content
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type", "TEXT") or "TEXT").upper()
                if part_type == "TEXT" and part.get("text"):
                    choice["text"] += part["text"]
                elif part_type == "THINKING" and part.get("thinking"):
                    choice["reasoning"] += part["thinking"]
        if message.get("reasoningContent"):
            choice["reasoning"] += message["reasoningContent"]
        self._accumulate_tool_calls(choice, message.get("toolCalls"))
        if payload.get("finishReason"):
            choice["finish_reason"] = payload["finishReason"]

    def _process_text(self, payload):
        """COHERE shaped events: ``text`` deltas, with the terminal event carrying the full text."""
        choice = self._choice(0)
        text = payload.get("text") or ""
        if payload.get("finishReason"):
            if text and len(text) >= len(choice["text"]):
                choice["text"] = text
            else:
                choice["text"] += text
            choice["finish_reason"] = payload["finishReason"]
            self._accumulate_tool_calls(choice, payload.get("toolCalls"))
        else:
            choice["text"] += text

    @staticmethod
    def _accumulate_tool_calls(choice, tool_calls):
        if not isinstance(tool_calls, list):
            return
        for position, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
            key = tool_call.get("id") or tool_call.get("index") or f"position-{position}"
            entry = choice["tool_calls"].setdefault(key, {"id": tool_call.get("id"), "name": "", "arguments": ""})
            if tool_call.get("id"):
                entry["id"] = tool_call["id"]
            name = tool_call.get("name") or function.get("name")
            if name:
                entry["name"] = name
            arguments = tool_call.get("arguments", function.get("arguments"))
            if arguments is None:
                arguments = tool_call.get("parameters")
            if isinstance(arguments, str):
                entry["arguments"] += arguments
            elif arguments is not None:
                entry["arguments"] = arguments

    def choices_list(self):
        """Choices ordered by index, with tool calls flattened to a list."""
        choices = []
        for index in sorted(self._choices):
            choice = self._choices[index]
            choices.append(
                {
                    "index": index,
                    "text": choice["text"],
                    "reasoning": choice["reasoning"],
                    "tool_calls": list(choice["tool_calls"].values()),
                    "finish_reason": choice["finish_reason"],
                }
            )
        return choices


class OCIGenAIStreamWrapper(ObjectProxy):
    """Proxy around the SDK's ``SSEClient`` that feeds events to the accumulator and finishes the span.

    ``on_done(accumulator, error, complete)`` is invoked exactly once: with ``complete=True`` when the stream was
    consumed to the end, with the exception when iteration failed, and with ``complete=False`` when the caller
    stopped early (``close()`` or leaving the ``events()`` loop before exhaustion).
    """

    def __init__(self, wrapped, accumulator, on_done, span=None):
        super().__init__(wrapped)
        self._self_accumulator = accumulator
        self._self_on_done = on_done
        self._self_span = span
        self._self_done = False

    def _span_scope(self):
        """Make the client span current while the SDK stream is read; never held across a ``yield``."""
        if self._self_span is None:
            return nullcontext()
        # Stream errors are recorded once by ``on_done``; keep ``use_span`` from recording them a second time.
        return use_span(self._self_span, end_on_exit=False, record_exception=False, set_status_on_exception=False)

    def events(self):
        try:
            iterator = self.__wrapped__.events()
            while True:
                with self._span_scope():
                    try:
                        event = next(iterator)
                    except StopIteration:
                        break
                    self._self_accumulator.process(getattr(event, "data", None))
                yield event
        except Exception as error:
            self._finish(error, complete=False)
            raise
        except BaseException:
            # ``GeneratorExit`` (the caller left the loop), ``KeyboardInterrupt``...: partial, not successful. The
            # caller is done with the stream, so release the SDK's event source instead of waiting for GC.
            self._finish(None, complete=False)
            self._close_wrapped()
            raise
        else:
            self._finish(None, complete=True)

    def close(self):
        """Close the underlying SSE stream; finishes the span as incomplete unless it was already consumed."""
        try:
            return self.__wrapped__.close()
        finally:
            self._finish(None, complete=False)

    def _close_wrapped(self):
        """Best-effort close of the SDK's ``SSEClient``; never masks the exception being propagated."""
        try:
            self.__wrapped__.close()
        except Exception as error:
            logger.debug("Failed to close the OCI GenAI SSE stream after cancellation: %s", error)

    def _finish(self, error, complete=True):
        if self._self_done:
            return
        self._self_done = True
        self._self_on_done(self._self_accumulator, error, complete)
