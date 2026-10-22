from opentelemetry.instrumentation.sagemaker.utils import dont_throw
from wrapt import ObjectProxy


class StreamingWrapper(ObjectProxy):
    def __init__(
        self,
        response,
        stream_done_callback=None,
    ):
        super().__init__(response)

        self._stream_done_callback = stream_done_callback
        self._accumulating_body = ""
        self._stream_done = False

    @property
    def accumulating_body(self):
        return self._accumulating_body

    def __iter__(self):
        # The finally block also runs when the caller breaks out of the loop
        # or drops the iterator, so the span ends with what was read so far.
        try:
            for event in self.__wrapped__:
                self._process_event(event)
                yield event
        except Exception as e:
            self._done(e)
            raise
        finally:
            self._done()

    def close(self):
        try:
            self.__wrapped__.close()
        finally:
            self._done()

    def _done(self, error=None):
        if self._stream_done:
            return
        self._stream_done = True
        self._stream_done_callback(self._accumulating_body, error)

    @dont_throw
    def _process_event(self, event):
        payload_part = event.get("PayloadPart")
        if not payload_part:
            return

        decoded_payload_part = payload_part.get("Bytes").decode()
        self._accumulating_body += decoded_payload_part
