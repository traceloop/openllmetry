import json
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv.ai import SpanAttributes
from opentelemetry.trace import set_span_in_context, Tracer
from opentelemetry.trace.span import Span

from opentelemetry import context as context_api
from opentelemetry.instrumentation.langchain.utils import (
    _with_tracer_wrapper,
)


class CustomJsonEncode(json.JSONEncoder):
    def default(self, o: Any) -> str:
        try:
            return super().default(o)
        except TypeError:
            return str(o)


@_with_tracer_wrapper
def callback_wrapper(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Hook into the invoke function, config is part of args, 2nd place.
    sources:
    https://python.langchain.com/v0.2/docs/how_to/callbacks_attach/
    https://python.langchain.com/v0.2/docs/how_to/callbacks_runtime/
    """
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)
    kind = to_wrap.get("kind")
    class_name = instance.get_name()
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
        cb.add_kind(class_name, kind)
        return wrapped(*args, **kwargs)
    else:
        cb.add_kind(class_name, kind)
        return wrapped(*args, {"callbacks": [cb, ]}, **kwargs)


class SyncSpanCallbackHandler(BaseCallbackHandler):
    def __init__(self, tracer: Tracer) -> None:
        self.tracer = tracer
        self.kinds = {}
        self.spans = {}

    def add_kind(self, class_name: str, kind: str) -> None:
        if class_name not in self.kinds:
            self.kinds[class_name] = (f"{class_name}.langchain.{kind}", kind)

    def _get_span(self, run_id: UUID):
        return self.spans[run_id]

    def _create_span(self, serialized: dict[str, Any], run_id: UUID) -> Span:
        name, kind = self.kinds[serialized["name"]]
        span = self.tracer.start_span(name)
        span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, kind)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name)
        self.spans[run_id] = span

        current_context = set_span_in_context(span)
        context_api.attach(current_context)

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
        span = self._create_span(serialized, run_id)
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            json.dumps({"inputs": inputs, "kwargs": kwargs}, cls=CustomJsonEncode),
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
        span.end()

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
        span = self._create_span(serialized, run_id)
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            json.dumps(
                {"input_str": input_str, "kwargs": kwargs}, cls=CustomJsonEncode,
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
        span.end()
