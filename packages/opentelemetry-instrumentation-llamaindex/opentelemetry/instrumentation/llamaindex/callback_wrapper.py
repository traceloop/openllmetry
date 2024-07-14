import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from opentelemetry import context as context_api
from opentelemetry.instrumentation.llamaindex.utils import (
    _with_tracer_wrapper,
    dont_throw,
    should_send_prompts,
)
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv.ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
    TraceloopSpanKindValues,
)
from opentelemetry.trace import set_span_in_context, Tracer
from opentelemetry.trace.span import Span


PARENT_ROOT = ("", "root")


class CustomJsonEncode(json.JSONEncoder):
    def default(self, o: Any) -> str:
        try:
            return super().default(o)
        except TypeError:
            return str(o)


@dataclass
class SpanHolder:
    span: Span
    token: Any
    context: context_api.context.Context
    children: list[str]


@_with_tracer_wrapper
def callback_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    callback_manager = kwargs.get("callback_manager") or instance.callback_manager
    if not any(isinstance(h, SyncSpanCallbackHandler) for h in callback_manager.handlers):
        callback_manager.add_handler(SyncSpanCallbackHandler(tracer))
    return wrapped(*args, **kwargs)


def _set_llm_request(span, payload) -> None:
    serialized = payload.get(EventPayload.SERIALIZED)
    span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, serialized.get("model"))
    span.set_attribute(SpanAttributes.LLM_REQUEST_TEMPERATURE, serialized.get("temperature"))
    if should_send_prompts():
        messages = payload.get(EventPayload.MESSAGES)
        for idx, message in enumerate(messages):
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{idx}.role", message.role)
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{idx}.content", message.content)


def _set_llm_response(span, payload) -> None:
    if should_send_prompts():
        messages = payload.get(EventPayload.MESSAGES)
        for idx, message in enumerate(messages):
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{idx}.role", message.role.value)
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{idx}.content", message.content)

        response = payload.get(EventPayload.RESPONSE)
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


class SyncSpanCallbackHandler(BaseCallbackHandler):
    def __init__(self, tracer: Tracer) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.tracer = tracer
        self.spans: dict[str, SpanHolder] = {}

    @staticmethod
    def _is_parent_root(parent_id) -> bool:
        return parent_id in PARENT_ROOT

    def _get_span(self, event_id: str) -> Span:
        return self.spans[event_id].span

    def _end_span(self, span: Span, event_id: str) -> None:
        for child_id in self.spans[event_id].children:
            child_span = self.spans[child_id].span
            if child_span.end_time is None:  # avoid warning on ended spans
                child_span.end()
        span.end()

    def _create_span(
        self,
        event_type: CBEventType,
        event_id: str = "",
        parent_id: str = "",
    ) -> Span:
        kind = (
            TraceloopSpanKindValues.WORKFLOW.value
            if self._is_parent_root(parent_id)
            else TraceloopSpanKindValues.TASK.value
        )
        span_name = f"{event_type.value}.llama_index.{kind}"

        if self._is_parent_root(parent_id):
            span = self.tracer.start_span(span_name)
        else:
            span = self.tracer.start_span(
                span_name, context=self.spans[parent_id].context, kind=kind
            )
            self.spans[parent_id].children.append(event_id)

        current_context = set_span_in_context(span)
        token = context_api.attach(
            context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True)
        )
        self.spans[event_id] = SpanHolder(span, token, current_context, [])

        span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, kind)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, span_name)

        return span

    @dont_throw
    def start_trace(self, trace_id: Optional[str] = None) -> None:
        return

    @dont_throw
    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        return

    @dont_throw
    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Run when an event starts and return id of event."""
        span = self._create_span(event_type, event_id, parent_id)
        if event_type == CBEventType.LLM:
            _set_llm_request(span, payload)
        elif should_send_prompts():
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                json.dumps(payload, cls=CustomJsonEncode),
            )
        return event_id

    @dont_throw
    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when an event ends."""
        span = self._get_span(event_id)
        if event_type == CBEventType.LLM:
            _set_llm_response(span, payload)
        elif should_send_prompts():
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                json.dumps(payload, cls=CustomJsonEncode),
            )
        self._end_span(span, event_id)
