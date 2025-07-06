import inspect
import json
import re
from dataclasses import dataclass, field
from functools import singledispatchmethod
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.agent import AgentToolCallEvent
from llama_index.core.instrumentation.events.chat_engine import (
    StreamChatEndEvent,
)
from llama_index.core.instrumentation.events.embedding import (
    EmbeddingStartEvent,
)
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
    LLMCompletionEndEvent,
    LLMPredictEndEvent,
)
from llama_index.core.instrumentation.events.rerank import ReRankStartEvent
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from llama_index.core.workflow import Workflow
from opentelemetry import context as context_api
from opentelemetry.instrumentation.llamaindex.event_emitter import (
    emit_chat_message_events,
    emit_chat_response_events,
    emit_rerank_message_event,
)
from opentelemetry.instrumentation.llamaindex.span_utils import (
    set_embedding,
    set_llm_chat_request,
    set_llm_chat_request_model_attributes,
    set_llm_chat_response,
    set_llm_chat_response_model_attributes,
    set_llm_predict_response,
    set_rerank,
    set_rerank_model_attributes,
    set_tool,
)
from opentelemetry.instrumentation.llamaindex.utils import (
    JSONEncoder,
    should_emit_events,
    should_send_prompts,
)
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
    TraceloopSpanKindValues,
)
from opentelemetry.trace import Span, Tracer, set_span_in_context

# For these spans, instead of creating a span using data from LlamaIndex,
# we use the regular OpenLLMetry instrumentations
AVAILABLE_OPENLLMETRY_INSTRUMENTATIONS = ["OpenAI"]

CLASS_ANDMETHOD_NAME_FROM_ID_REGEX = re.compile(r"([a-zA-Z]+)\.([a-zA-Z_]+)-")
STREAMING_END_EVENTS = (
    LLMChatEndEvent,
    LLMCompletionEndEvent,
    StreamChatEndEvent,
)


def instrument_with_dispatcher(tracer: Tracer):
    dispatcher = get_dispatcher()
    openllmetry_span_handler = OpenLLMetrySpanHandler(tracer)
    dispatcher.add_span_handler(openllmetry_span_handler)
    dispatcher.add_event_handler(OpenLLMetryEventHandler(openllmetry_span_handler))


@dataclass
class SpanHolder:
    span_id: str
    parent: Optional["SpanHolder"] = None
    otel_span: Optional[Span] = None
    token: Optional[Any] = None
    context: Optional[context_api.context.Context] = None
    waiting_for_streaming: bool = field(init=False, default=False)

    _active: bool = field(init=False, default=True)

    def process_event(self, event: BaseEvent) -> List["SpanHolder"]:
        self.update_span_for_event(event)

        if self.waiting_for_streaming and isinstance(event, STREAMING_END_EVENTS):
            self.end()
            return [self] + self.notify_parent()

        return []

    def notify_parent(self) -> List["SpanHolder"]:
        if self.parent:
            self.parent.end()
            return [self.parent] + self.parent.notify_parent()
        return []

    def end(self, should_detach_context: bool = True):
        if not self._active:
            return

        self._active = False
        if self.otel_span:
            self.otel_span.end()
        if self.token and should_detach_context:
            context_api.detach(self.token)

    @singledispatchmethod
    def update_span_for_event(self, event: BaseEvent):
        pass

    @update_span_for_event.register
    def _(self, event: LLMChatStartEvent):
        set_llm_chat_request_model_attributes(event, self.otel_span)
        if should_emit_events():
            emit_chat_message_events(event)
        else:
            set_llm_chat_request(event, self.otel_span)

    @update_span_for_event.register
    def _(self, event: LLMChatEndEvent):
        set_llm_chat_response_model_attributes(event, self.otel_span)
        if should_emit_events():
            emit_chat_response_events(event)
        else:
            set_llm_chat_response(event, self.otel_span)  # noqa: F821

    @update_span_for_event.register
    def _(self, event: LLMPredictEndEvent):
        if not should_emit_events():
            set_llm_predict_response(event, self.otel_span)

    @update_span_for_event.register
    def _(self, event: EmbeddingStartEvent):
        set_embedding(event, self.otel_span)

    @update_span_for_event.register
    def _(self, event: ReRankStartEvent):
        set_rerank_model_attributes(event, self.otel_span)
        if should_emit_events():
            emit_rerank_message_event(event)
        else:
            set_rerank(event, self.otel_span)

    @update_span_for_event.register
    def _(self, event: AgentToolCallEvent):
        set_tool(event, self.otel_span)


class OpenLLMetrySpanHandler(BaseSpanHandler[SpanHolder]):
    waiting_for_streaming_spans: Dict[str, SpanHolder] = {}
    _tracer: Tracer = PrivateAttr()

    def __init__(self, tracer: Tracer):
        super().__init__()
        self._tracer = tracer

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[SpanHolder]:
        """Create a span."""
        # Take the class name and method name from id_ where id_ is e.g.
        # 'SentenceSplitter.split_text_metadata_aware-a2f2a780-2fa6-4682-a88e-80dc1f1ebe6a'
        matches = CLASS_ANDMETHOD_NAME_FROM_ID_REGEX.match(id_)
        class_name = matches.groups()[0]
        method_name = matches.groups()[1]

        parent = self.open_spans.get(parent_span_id)

        if class_name in AVAILABLE_OPENLLMETRY_INSTRUMENTATIONS:
            token = context_api.attach(
                context_api.set_value(
                    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, False
                )
            )
            return SpanHolder(id_, parent, token=token)

        kind = (
            TraceloopSpanKindValues.TASK.value
            if parent
            else TraceloopSpanKindValues.WORKFLOW.value
        )

        if isinstance(instance, Workflow):
            span_name = (
                f"{instance.__class__.__name__}.{kind}"
                if not parent_span_id
                else f"{method_name}.{kind}"
            )
        else:
            span_name = f"{class_name}.{kind}"

        span = self._tracer.start_span(
            span_name,
            context=parent.context if parent else None,
        )
        current_context = set_span_in_context(
            span, context=parent.context if parent else None
        )
        current_context = context_api.set_value(
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
            True,
            current_context,
        )
        token = context_api.attach(current_context)

        span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, kind)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, span_name)
        try:
            if should_send_prompts():
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_INPUT,
                    json.dumps(bound_args.arguments, cls=JSONEncoder),
                )
        except Exception:
            pass

        return SpanHolder(id_, parent, span, token, current_context)

    def prepare_to_exit_span(
        self,
        id_: str,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs,
    ) -> SpanHolder:
        """Logic for preparing to drop a span."""
        span_holder = self.open_spans[id_]
        # I know it's messy, but the typing of result is messy and couldn't find a better way
        # to get a dictionary I can then use to remove keys
        try:
            serialized_output = json.dumps(result, cls=JSONEncoder)
            # we need to remove some keys like source_nodes as they can be very large
            output = json.loads(serialized_output)
            if "source_nodes" in output:
                del output["source_nodes"]
            if should_send_prompts():
                span_holder.otel_span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                    json.dumps(output, cls=JSONEncoder),
                )
        except Exception:
            pass

        if isinstance(result, (Generator, AsyncGenerator, StreamingResponse)):
            # This is a streaming response, we want to wait for the streaming end event before ending the span
            span_holder.waiting_for_streaming = True
            with self.lock:
                self.waiting_for_streaming_spans[id_] = span_holder
            return span_holder
        else:
            should_detach_context = not isinstance(instance, Workflow)
            span_holder.end(should_detach_context)
            return span_holder

    def prepare_to_drop_span(
        self, id_: str, err: Optional[Exception], **kwargs
    ) -> Optional[SpanHolder]:
        """Logic for dropping a span."""
        if id_ in self.open_spans:
            with self.lock:
                span_holder = self.open_spans[id_]
            return span_holder
        return None


class OpenLLMetryEventHandler(BaseEventHandler):
    _span_handler: OpenLLMetrySpanHandler = PrivateAttr()

    def __init__(self, span_handler: OpenLLMetrySpanHandler):
        super().__init__()
        self._span_handler = span_handler

    def handle(self, event: BaseEvent, **kwargs) -> Any:
        span = self._span_handler.open_spans.get(event.span_id)
        if not span:
            span = self._span_handler.waiting_for_streaming_spans.get(event.span_id)
        if not span:
            print(f"No span found for event {event}")
            return

        finished_spans = span.process_event(event)

        with self._span_handler.lock:
            for span in finished_spans:
                self._span_handler.waiting_for_streaming_spans.pop(span.span_id)
