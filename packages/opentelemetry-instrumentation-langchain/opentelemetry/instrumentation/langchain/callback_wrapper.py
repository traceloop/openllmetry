import json
from typing import Any, Dict

from langchain_core.callbacks import BaseCallbackHandler
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


def get_name(to_wrap, instance) -> str:
    return f"{instance.get_name()}.langchain.{to_wrap.get('kind')}"


def get_kind(to_wrap) -> str:
    return to_wrap.get("kind")


@_with_tracer_wrapper
def callback_wrapper(tracer, to_wrap, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)
    kind = get_kind(to_wrap)
    name = get_name(to_wrap, instance)
    cb = SyncSpanCallbackHandler(tracer, name, kind)
    if "callbacks" in kwargs:
        if not any(isinstance(c, SyncSpanCallbackHandler) for c in kwargs["callbacks"]):
            # Avoid adding the same callback twice, e.g. SequentialChain is also a Chain
            kwargs["callbacks"].append(cb)
    else:
        kwargs["callbacks"] = [
            cb,
        ]
    return wrapped(*args, **kwargs)


class SyncSpanCallbackHandler(BaseCallbackHandler):
    def __init__(self, tracer: Tracer, name: str, kind: str) -> None:
        self.tracer = tracer
        self.name = name
        self.kind = kind
        self.span: Span

    def _create_span(self) -> None:
        self.span = self.tracer.start_span(self.name)
        self.span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, self.kind)
        self.span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, self.name)

        current_context = set_span_in_context(self.span)
        context_api.attach(current_context)

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        self._create_span()
        self.span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            json.dumps({"inputs": inputs, "kwargs": kwargs}, cls=CustomJsonEncode),
        )

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        self.span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            json.dumps({"outputs": outputs, "kwargs": kwargs}, cls=CustomJsonEncode),
        )
        self.span.end()

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""
        self._create_span()
        self.span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            json.dumps(
                {"input_str": input_str, "kwargs": kwargs}, cls=CustomJsonEncode
            ),
        )

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Run when tool ends running."""
        self.span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            json.dumps({"output": output, "kwargs": kwargs}, cls=CustomJsonEncode),
        )
        self.span.end()
