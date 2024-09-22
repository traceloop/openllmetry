import json
from opentelemetry.instrumentation.bedrock.utils import dont_throw
from wrapt import ObjectProxy


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
        for event in self.__wrapped__:
            self._process_event(event)
            yield event

    @dont_throw
    def _process_event(self, event):
        chunk = event.get("chunk")
        if not chunk:
            return

        decoded_chunk = json.loads(chunk.get("bytes").decode())
        type = decoded_chunk.get("type")

        if type == "message_start":
            self._accumulating_body = decoded_chunk.get("message")
        elif type == "content_block_start":
            self._accumulating_body["content"].append(
                decoded_chunk.get("content_block")
            )
        elif type == "content_block_delta":
            self._accumulating_body["content"][-1]["text"] += decoded_chunk.get(
                "delta"
            ).get("text")
        elif type == "message_stop" and self._stream_done_callback:
            self._accumulating_body["invocation_metrics"] = decoded_chunk.get("amazon-bedrock-invocationMetrics")
            self._stream_done_callback(self._accumulating_body)
