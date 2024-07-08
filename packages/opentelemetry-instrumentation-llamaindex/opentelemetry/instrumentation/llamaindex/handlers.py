from inspect import BoundArguments
from typing import Any
from llama_index.core.instrumentation.span import BaseSpan
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from opentelemetry.semconv.ai import SpanAttributes, TraceloopSpanKindValues
from opentelemetry.trace import (
    Context,
    Span as OtelSpan,
    Tracer,
    context_api,
    set_span_in_context,
)


class _Span(BaseSpan, extra="allow"):
    _otel_span: OtelSpan
    _span_name: str
    _kind: TraceloopSpanKindValues
    context: Context
    token: object

    def __init__(
        self,
        name: str,
        otel_span: OtelSpan,
        kind: TraceloopSpanKindValues,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        otel_span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, kind.value)
        otel_span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name)

        self.token = context_api.attach(set_span_in_context(otel_span))

        self._kind = kind
        self._span_name = name
        self._otel_span = otel_span

    def context(self) -> Context:
        return set_span_in_context(
            self._otel_span, context_api.set_value("workflow_name", self._span_name)
        )

    def end(self):
        self._otel_span.end()
        context_api.detach(self.token)


class SpanHandler(BaseSpanHandler[_Span], extra="allow"):
    _tracer: Tracer

    def __init__(self, tracer: Tracer):
        super().__init__()

        self._tracer = tracer

    def new_span(
        self,
        id_: str,
        bound_args: BoundArguments,
        instance: Any | None = None,
        parent_span_id: str | None = None,
        **kwargs: Any,
    ) -> _Span | None:
        with self.lock:
            parent = self.open_spans.get(parent_span_id)

            kind = (
                TraceloopSpanKindValues.WORKFLOW
                if parent_span_id is None
                else TraceloopSpanKindValues.TASK
            )
            # id_ is like BaseQueryEngine.aquery-d4e9e25e-7852-40ab-bb56-ada5fb679ff0
            name = (
                f"{id_.partition('.')[0].replace('Base', '')}.llamaindex.{kind.value}"
            )

            otel_span = self._tracer.start_span(
                name, context=(parent.context if parent else None)
            )

        span = _Span(
            name=name,
            otel_span=otel_span,
            kind=kind,
            id_=id_,
            parent_span_id=parent_span_id,
        )

        return span

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: BoundArguments,
        instance: Any | None = None,
        result: Any | None = None,
        **kwargs: Any,
    ) -> _Span | None:
        with self.lock:
            span = self.open_spans[id_]

        if span:
            span.end()
            return span

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: BoundArguments,
        instance: Any | None = None,
        err: BaseException | None = None,
        **kwargs: Any,
    ) -> _Span | None:
        return super().prepare_to_drop_span(id_, bound_args, instance, err, **kwargs)


class EventHandler(BaseEventHandler, extra="allow"):
    _span_handler: SpanHandler

    def __init__(self, span_handler: SpanHandler):
        super().__init__()
        self._span_handler = span_handler

    def handle(self, event: BaseEvent, **kwargs) -> Any:
        return super().handle(event, **kwargs)
