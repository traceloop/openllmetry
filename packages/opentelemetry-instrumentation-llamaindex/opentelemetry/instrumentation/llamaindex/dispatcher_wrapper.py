import re
from dataclasses import dataclass
from typing import Any, Optional

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.llm import LLMChatEndEvent, LLMChatStartEvent
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from opentelemetry import context as context_api
from opentelemetry.instrumentation.llamaindex.utils import (
    dont_throw,
    should_send_prompts,
)
from opentelemetry.semconv.ai import (
    SpanAttributes,
    TraceloopSpanKindValues,
)
from opentelemetry.trace import get_current_span, set_span_in_context, Tracer
from opentelemetry.trace.span import Span


LLAMA_INDEX_REGEX = re.compile(r"^([a-zA-Z]+)\.")


def instrument_with_dispatcher(tracer: Tracer):
    dispatcher = get_dispatcher()
    openll_span_handler = OpenLLSpanHandler(tracer)
    dispatcher.add_span_handler(openll_span_handler)
    dispatcher.add_event_handler(OpenLLEventHandler(openll_span_handler))


@dont_throw
def _set_llm_request(event, span) -> None:
    model_dict = event.model_dict
    span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model_dict.get("model"))
    span.set_attribute(SpanAttributes.LLM_REQUEST_TEMPERATURE, model_dict.get("temperature"))
    if should_send_prompts():
        for idx, message in enumerate(event.messages):
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{idx}.role", message.role.value)
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{idx}.content", message.content)


@dont_throw
def _set_llm_response(event, span) -> None:
    if should_send_prompts():
        for idx, message in enumerate(event.messages):
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{idx}.role", message.role.value)
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{idx}.content", message.content)
        response = event.response
        span.set_attribute(
            f"{SpanAttributes.LLM_COMPLETIONS}.role",
            response.message.role.value,
        )
        span.set_attribute(
            f"{SpanAttributes.LLM_COMPLETIONS}.content",
            response.message.content,
        )
        if not (raw := response.raw):
            return
        span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, raw.get("model"))
        if usage := response.raw.get("usage"):
            span.set_attribute(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, usage.completion_tokens)
            span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.prompt_tokens)
            span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.total_tokens)


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
        # Take the class name from id_ where id_ is e.g.
        # 'SentenceSplitter.split_text_metadata_aware-a2f2a780-2fa6-4682-a88e-80dc1f1ebe6a'
        class_name = LLAMA_INDEX_REGEX.match(id_).groups()[0]
        span_name = f"{class_name}.llama_index.{kind}"
        span = self._tracer.start_span(
            span_name,
            context=parent.context if parent else None,
        )
        current_context = set_span_in_context(span, context=parent.context if parent else None)
        token = context_api.attach(current_context)
        span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, kind)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, span_name)
        return SpanHolder(span, token, current_context)

    def prepare_to_exit_span(
        self, id_: str, result: Optional[Any] = None, **kwargs
    ) -> SpanHolder:
        """Logic for preparing to drop a span."""
        span_holder = self.open_spans[id_]
        span_holder.span.end()
        context_api.detach(span_holder.token)
        with self.lock:
            self.completed_spans += [span_holder]
        return span_holder

    def prepare_to_drop_span(
        self, id_: str, err: Optional[Exception], **kwargs
    ) -> Optional[SpanHolder]:
        """Logic for dropping a span."""
        if id_ in self.open_spans:
            with self.lock:
                span_holder = self.open_spans[id_]
                self.dropped_spans += [span_holder]
            return span_holder
        return None


class OpenLLEventHandler(BaseEventHandler):
    _span_handler: OpenLLSpanHandler = PrivateAttr()

    def __init__(self, span_handler: OpenLLSpanHandler):
        super().__init__()
        self._span_handler = span_handler

    def handle(self, event: BaseEvent, **kwargs) -> Any:
        if isinstance(event, LLMChatStartEvent):
            span = get_current_span()
            _set_llm_request(event, span)
        elif isinstance(event, LLMChatEndEvent):
            span = get_current_span()
            _set_llm_response(event, span)
