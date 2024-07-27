import re
from dataclasses import dataclass
from typing import Any, Optional

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.span.base import BaseSpan
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from opentelemetry import context as context_api
from opentelemetry.semconv.ai import (
    SpanAttributes,
    TraceloopSpanKindValues,
)
from opentelemetry.trace import set_span_in_context, Tracer
from opentelemetry.trace.span import Span


LLAMA_INDEX_REGEX = re.compile(r"^([a-zA-Z]+)\.")


def instrument_with_dispatcher(tracer: Tracer):
    dispatcher = get_dispatcher()
    dispatcher.add_span_handler(OpenLLSpanHandler(tracer))


@dataclass
class SpanHolder:
    span: Span
    token: Any
    context: context_api.context.Context


class OpenLLSpanHandler(BaseSpanHandler[SpanHolder]):
    _tracer: Tracer = PrivateAttr()

    def __init__(self, tracer: Tracer):
        super().__init__()
        self._tracer = tracer

    def new_span(
        self, id_: str, parent_span_id: Optional[str], **kwargs
    ) -> Optional[SpanHolder]:
        """Create a span."""
        parent = self.open_spans.get(parent_span_id)
        kind = (
            TraceloopSpanKindValues.TASK.value
            if parent
            else TraceloopSpanKindValues.WORKFLOW.value
        )
        class_name = LLAMA_INDEX_REGEX.match(id_).groups()[0]
        span_name = f"{class_name}.llama_index.{kind}"

        if parent:
            span = self._tracer.start_span(
                span_name, context=parent.context, kind=kind
            )
        else:
            span = self._tracer.start_span(span_name)

        current_context = set_span_in_context(span)
        token = context_api.attach(current_context)
        span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, kind)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, span_name)

        return SpanHolder(span, token, current_context)

    def prepare_to_exit_span(
        self, id_: str, result: Optional[Any] = None, **kwargs
    ) -> Any:
        """Logic for preparing to exit a span."""
        with self.lock:
            span_holder = self.open_spans[id_]
        span_holder.span.end()

    def prepare_to_drop_span(
        self, id_: str, err: Optional[Exception], **kwargs
    ) -> Any:
        """Logic for preparing to drop a span."""
        pass
