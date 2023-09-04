import os

from opentelemetry.context import (
    attach,
    set_value,
    _SUPPRESS_INSTRUMENTATION_KEY,
    detach,
)
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import logging

logger = logging.getLogger(__name__)


class NoLogSpanBatchProcessor(BatchSpanProcessor):
    def _export_batch(self) -> int:
        """Exports at most max_export_batch_size spans and returns the number of
        exported spans.
        """
        idx = 0
        # currently only a single thread acts as consumer, so queue.pop() will
        # not raise an exception
        while idx < self.max_export_batch_size and self.queue:
            self.spans_list[idx] = self.queue.pop()
            idx += 1
        token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        try:
            # Ignore type b/c the Optional[None]+slicing is too "clever"
            # for mypy
            self.span_exporter.export(self.spans_list[:idx])  # type: ignore
        except Exception:  # pylint: disable=broad-except
            if os.getenv("TRACELOOP_LOGGING_ENABLED", "false").lower() == "true":
                logger.exception("Exception while exporting Span batch.")
        detach(token)

        # clean up list
        for index in range(idx):
            self.spans_list[index] = None
        return idx
