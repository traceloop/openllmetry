import inspect
import json
import re
from typing import Any, Dict, Optional
from dataclasses import dataclass

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.agent import AgentToolCallEvent
from llama_index.core.instrumentation.events.embedding import EmbeddingStartEvent
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
    LLMPredictEndEvent,
)
from llama_index.core.instrumentation.events.rerank import ReRankStartEvent
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from opentelemetry import context as context_api
from opentelemetry.instrumentation.llamaindex.utils import (
    JSONEncoder,
    dont_throw,
    should_send_prompts,
)
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
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
def _set_llm_chat_request(event, span) -> None:
    model_dict = event.model_dict
    span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model_dict.get("model"))
    span.set_attribute(
        SpanAttributes.LLM_REQUEST_TEMPERATURE, model_dict.get("temperature")
    )
    if should_send_prompts():
        for idx, message in enumerate(event.messages):
            span.set_attribute(
                f"{SpanAttributes.LLM_PROMPTS}.{idx}.role", message.role.value
            )
            span.set_attribute(
                f"{SpanAttributes.LLM_PROMPTS}.{idx}.content", message.content
            )


@dont_throw
def _set_llm_chat_response(event, span) -> None:
    response = event.response
    if should_send_prompts():
        for idx, message in enumerate(event.messages):
            span.set_attribute(
                f"{SpanAttributes.LLM_PROMPTS}.{idx}.role", message.role.value
            )
            span.set_attribute(
                f"{SpanAttributes.LLM_PROMPTS}.{idx}.content", message.content
            )
        span.set_attribute(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.role",
            response.message.role.value,
        )
        span.set_attribute(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
            response.message.content,
        )
    if not (raw := response.raw):
        return
    span.set_attribute(
        SpanAttributes.LLM_RESPONSE_MODEL,
        raw.get("model") if "model" in raw else raw.model,  # raw can be Any, not just ChatCompletion
    )
    if usage := raw.get("usage") if "usage" in raw else raw.usage:
        span.set_attribute(
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, usage.completion_tokens
        )
        span.set_attribute(
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.prompt_tokens
        )
        span.set_attribute(
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.total_tokens
        )
    if choices := raw.choices:
        span.set_attribute(
            SpanAttributes.LLM_RESPONSE_FINISH_REASON, choices[0].finish_reason
        )


@dont_throw
def _set_llm_predict_response(event, span) -> None:
    if should_send_prompts():
        span.set_attribute(
            f"{SpanAttributes.LLM_COMPLETIONS}.role",
            MessageRole.ASSISTANT.value,
        )
        span.set_attribute(
            f"{SpanAttributes.LLM_COMPLETIONS}.content",
            event.output,
        )


@dont_throw
def _set_embedding(event, span) -> None:
    model_dict = event.model_dict
    span.set_attribute(
        f"{LLMRequestTypeValues.EMBEDDING.value}.model_name",
        model_dict.get("model_name"),
    )


@dont_throw
def _set_rerank(event, span) -> None:
    span.set_attribute(
        f"{LLMRequestTypeValues.RERANK.value}.model_name",
        event.model_name,
    )
    span.set_attribute(
        f"{LLMRequestTypeValues.RERANK.value}.top_n",
        event.top_n,
    )
    if should_send_prompts():
        span.set_attribute(
            f"{LLMRequestTypeValues.RERANK.value}.query",
            event.query.query_str,
        )


@dont_throw
def _set_tool(event, span) -> None:
    span.set_attribute("tool.name", event.tool.name)
    span.set_attribute("tool.description", event.tool.description)
    span.set_attribute("tool.arguments", event.arguments)


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
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
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
                    json.dumps(bound_args.arguments, cls=JSONEncoder)
                )
        except Exception:
            pass

        return SpanHolder(span, token, current_context)

    def prepare_to_exit_span(
        self, id_: str, result: Optional[Any] = None, **kwargs
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
                span_holder.span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                    json.dumps(output, cls=JSONEncoder),
                )
        except Exception:
            pass

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
        span = get_current_span()
        # use case with class_pattern if support for 3.9 is dropped
        if isinstance(event, LLMChatStartEvent):
            _set_llm_chat_request(event, span)
        elif isinstance(event, LLMChatEndEvent):
            _set_llm_chat_response(event, span)
        elif isinstance(event, LLMPredictEndEvent):
            _set_llm_predict_response(event, span)
        elif isinstance(event, EmbeddingStartEvent):
            _set_embedding(event, span)
        elif isinstance(event, ReRankStartEvent):
            _set_rerank(event, span)
        elif isinstance(event, AgentToolCallEvent):
            _set_tool(event, span)
