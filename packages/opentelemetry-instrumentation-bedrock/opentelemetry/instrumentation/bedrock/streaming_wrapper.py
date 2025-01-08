import json
from opentelemetry.instrumentation.bedrock.utils import dont_throw
from wrapt import ObjectProxy
from opentelemetry.instrumentation.bedrock.config import Config
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Span 
import logging

logger = logging.getLogger(__name__)

class StreamingWrapper(ObjectProxy):
    def __init__(
        self,
        response,
        stream_done_callback=None,
        span: Span = None
    ):
        super().__init__(response)
        self._stream_done_callback = stream_done_callback
        self._accumulating_body = {}
        self._span = span

    def __iter__(self):
        for event in self.__wrapped__:
            self._process_event(event)
            yield event

    @dont_throw
    def _process_event(self, event):
        chunk = event.get("chunk")
        if not chunk:
            return

        decoded_chunk = json.loads(chunk.get("bytes").decode())

        # Check if the response is from an Anthropic model (has a "type" field)
        if "type" in decoded_chunk:
            # Anthropic model streaming logic
            type = decoded_chunk.get("type")
            logger.debug(f"Received streaming event of type: {type}")
            logger.debug(f"Decoded chunk: {decoded_chunk}")

            if type == "message_start":
                self._accumulating_body = decoded_chunk.get("message")
                if not Config.use_legacy_attributes and self._span and self._accumulating_body.get("role") == "user":
                    # Initialize content for prompt event to empty string
                    _emit_prompt_event_streaming(self._span, self._accumulating_body.get("role"), "", 0)

            elif type == "content_block_start":
                self._accumulating_body["content"].append(decoded_chunk.get("content_block"))

            elif type == "content_block_delta":
                if not self._accumulating_body.get("content"):
                    self._accumulating_body["content"] = [{"text": ""}]

                # Accumulate content for both legacy attributes and new events
                self._accumulating_body["content"][-1]["text"] += decoded_chunk.get("delta").get("text")

                if not Config.use_legacy_attributes and self._span:
                    if self._accumulating_body.get("role") == "user":
                        # Update content for prompt event
                        _emit_prompt_event_streaming(
                            self._span,
                            self._accumulating_body.get("role"),
                            self._accumulating_body["content"][-1]["text"],
                            0,
                        )
                    elif self._accumulating_body.get("role") == "assistant":
                        # Emit completion event delta
                        _emit_completion_event_streaming(
                            self._span, decoded_chunk.get("delta").get("text"), 0
                        )

        else:
            # Amazon Titan model streaming logic
            logger.debug(f"Received streaming event: {decoded_chunk}")

            if not Config.use_legacy_attributes and self._span and decoded_chunk.get("outputText"):
                # Emit completion event for each chunk of text
                _emit_completion_event_streaming(
                    self._span, decoded_chunk.get("outputText", ""), 0
                )

            # Accumulate the response for the final callback
            if not hasattr(self, "_accumulated_amazon_output"):
                self._accumulated_amazon_output = ""
            self._accumulated_amazon_output += decoded_chunk.get("outputText", "")

            # Check if this is the end of the stream
            if decoded_chunk.get("completionReason") == "FINISH" and self._stream_done_callback:
                self._stream_done_callback({
                    "text": self._accumulated_amazon_output,
                    "role": "assistant"
                })

    def _get_accumulated_content_text(self):
        """Helper function to extract accumulated text from content blocks."""
        return "".join(block.get("text", "") for block in self._accumulating_body.get("content", []))

def _emit_prompt_event_streaming(span: Span, role: str, content: str, index: int):
    """Emit a prompt event for streaming responses."""
    attributes = {
        "messaging.role": role,
        "messaging.content": content,
        "messaging.index": index,
    }
    span.add_event("prompt", attributes=attributes)

def _emit_completion_event_streaming(span: Span, content: str, index: int):
    """Emit a completion event for streaming responses."""
    attributes = {
        "messaging.content": content,
        "messaging.index": index,
    }
    span.add_event("completion", attributes=attributes)