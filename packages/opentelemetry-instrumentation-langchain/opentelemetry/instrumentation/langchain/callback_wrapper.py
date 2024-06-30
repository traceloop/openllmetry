import json
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager
from langchain_core.messages import BaseMessage
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv.ai import LLMRequestTypeValues, SpanAttributes, TraceloopSpanKindValues
from opentelemetry.context.context import Context
from opentelemetry.trace import set_span_in_context, Tracer
from opentelemetry.trace.span import Span

from opentelemetry import context as context_api
from opentelemetry.instrumentation.langchain.utils import (
    _with_tracer_wrapper,
    dont_throw,
)


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
    context: Context
    children: list[UUID]


@dont_throw
def _add_callback(tracer, to_wrap, instance, args):
    cb = SyncSpanCallbackHandler(tracer)
    if len(args) > 1:
        if "callbacks" in args[1]:
            temp_list = args[1]["callbacks"]
            if isinstance(temp_list, BaseCallbackManager):
                for c in temp_list.handlers:
                    if isinstance(c, SyncSpanCallbackHandler):
                        cb = c
                        break
                else:
                    args[1]["callbacks"].add_handler(cb)
            elif isinstance(temp_list, list):
                for c in temp_list:
                    if isinstance(c, SyncSpanCallbackHandler):
                        cb = c
                        break
                    else:
                        args[1]["callbacks"].append(cb)
        else:
            args[1].update({"callbacks": [cb, ]})
    return cb


@_with_tracer_wrapper
def callback_wrapper(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Hook into the invoke function, config is part of args, 2nd place.
    sources:
    https://python.langchain.com/v0.2/docs/how_to/callbacks_attach/
    https://python.langchain.com/v0.2/docs/how_to/callbacks_runtime/
    """
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)
    cb = _add_callback(tracer, to_wrap, instance, args)
    if len(args) > 1:
        return wrapped(*args, **kwargs)
    return wrapped(*args, {"callbacks": [cb, ]}, **kwargs)


def _set_chat_request(span: Span, serialized: Any) -> None:
    span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.CHAT.value)
    try:
        kwargs = serialized["kwargs"]
        for model_tag in ("model", "model_id", "model_name"):
            if (model := kwargs.get(model_tag)) is not None:
                break
        else:
            model = "unknown"
        span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model)
    except KeyError:
        pass


class SyncSpanCallbackHandler(BaseCallbackHandler):
    def __init__(self, tracer: Tracer) -> None:
        self.tracer = tracer
        self.spans: dict[UUID, SpanHolder] = {}

    @staticmethod
    def _get_name_from_callback(
        serialized: dict[str, Any],
        _tags: Optional[list[str]] = None,
        _metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Get the name to be used for the span. Based on heuristic. Can be extended."""
        try:
            return serialized["kwargs"]["name"]
        except KeyError:
            pass
        try:
            return kwargs["name"]
        except KeyError:
            return serialized["id"][-1]

    def _get_span(self, run_id: UUID) -> Span:
        return self.spans[run_id].span

    def _end_span(self, span: Span, run_id: UUID) -> None:
        for child_id in self.spans[run_id].children:
            child_span = self.spans[child_id].span
            if child_span.end_time is None:  # avoid warning on ended spans
                child_span.end()
        span.end()

    def _create_span(self, run_id: UUID, parent_run_id: Optional[UUID], name: str) -> Span:
        kind = TraceloopSpanKindValues.WORKFLOW.value if parent_run_id is None else TraceloopSpanKindValues.TASK.value
        span_name = f"{name}.langchain.{kind}"
        if parent_run_id is not None:
            span = self.tracer.start_span(span_name, context=self.spans[parent_run_id].context)
        else:
            span = self.tracer.start_span(span_name)
        span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, kind)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, span_name)
        current_context = set_span_in_context(span)
        token = context_api.attach(current_context)
        self.spans[run_id] = SpanHolder(span, token, current_context, [])
        if parent_run_id is not None:
            self.spans[parent_run_id].children.append(run_id)
        return span

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain starts running."""
        name = self._get_name_from_callback(serialized, **kwargs)
        span = self._create_span(run_id, parent_run_id, name)
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            json.dumps(
                {
                    "inputs": inputs,
                    "tags": tags,
                    "metadata": metadata,
                    "kwargs": kwargs,
                },
                cls=CustomJsonEncode,
            ),
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain ends running."""
        span = self._get_span(run_id)
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            json.dumps({"outputs": outputs, "kwargs": kwargs}, cls=CustomJsonEncode),
        )
        self._end_span(span, run_id)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Chat Model starts running."""
        name = self._get_name_from_callback(serialized, kwargs=kwargs)
        span = self._create_span(run_id, parent_run_id, name)
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            json.dumps(
                {
                    "messages": messages,
                    "metadata": metadata,
                    "name": name,
                    "kwargs": kwargs,
                },
                cls=CustomJsonEncode,
            ),
        )
        _set_chat_request(span, serialized)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool starts running."""
        name = self._get_name_from_callback(serialized, kwargs=kwargs)
        span = self._create_span(run_id, parent_run_id, name)
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            json.dumps(
                {
                    "input_str": input_str,
                    "tags": tags,
                    "metadata": metadata,
                    "inputs": inputs,
                    "kwargs": kwargs,
                },
                cls=CustomJsonEncode,
            ),
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running."""
        span = self._get_span(run_id)
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            json.dumps({"output": output, "kwargs": kwargs}, cls=CustomJsonEncode),
        )
        self._end_span(span, run_id)
