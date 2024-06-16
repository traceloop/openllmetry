import json
from typing import Any, Dict

from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv.ai import SpanAttributes
from opentelemetry.trace import set_span_in_context, Tracer
from opentelemetry.trace.span import Span

from opentelemetry import context as context_api
from opentelemetry.instrumentation.langchain.utils import (
    _with_tracer_wrapper,
)


TAG_PREFIX = "tag_openllmetry"


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
    """Hook into the invoke function. Note: config is part of args, 2nd place.
    sources:
    https://python.langchain.com/v0.2/docs/how_to/callbacks_attach/
    https://python.langchain.com/v0.2/docs/how_to/callbacks_runtime/
    """
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)
    # Add tag
    if instance.tags is None:
        instance.tags = []
    instance.tags.append(f"{TAG_PREFIX}{id(instance)}")
    kind = get_kind(to_wrap)
    name = get_name(to_wrap, instance)
    cb = SyncSpanCallbackHandler(tracer)
    if len(args) > 1:
        if "callbacks" in args[1]:
            temp_list = args[1]["callbacks"]
            #if isinstance(temp_list, list):
            #    if not any(isinstance(c, SyncSpanCallbackHandler) for c in temp_list):
            #        args[1]["callbacks"].append(cb)
            if isinstance(temp_list, BaseCallbackManager):
                for c in temp_list.handlers:
                    if isinstance(c, SyncSpanCallbackHandler):
                        cb = c
                        break
                else:
                    args[1]["callbacks"].add_handler(cb)
        else:
            args[1].update({"callbacks": [cb, ]})
        cb.add_handler(id(instance), name, kind)
        return wrapped(*args, **kwargs)
    else:
        cb.add_handler(id(instance), name, kind)
        return wrapped(*args, {"callbacks": [cb, ]}, **kwargs)


class SyncSpanCallbackHandler(BaseCallbackHandler):
    def __init__(self, tracer: Tracer) -> None:
        self.tracer = tracer
        self.handlers = {}
        self.spans = {}

    @staticmethod
    def _get_handler_id(tags: list[str]) -> int:
        for tag in tags:
            if tag.startswith(TAG_PREFIX):
                return int(tag.removeprefix(TAG_PREFIX))
        raise RuntimeError

    def _get_span(self, handler_id) -> Span:
        return self.spans[handler_id]

    def _create_span(self, handler_id) -> Span:
        handler_params = self.handlers[handler_id]
        span = self.tracer.start_span(handler_params[0])
        span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, handler_params[1])
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, handler_params[0])
        self.spans[handler_id] = span

        current_context = set_span_in_context(span)
        context_api.attach(current_context)

        return span

    def add_handler(self, handler_id: int, name: str, kind: str) -> None:
        self.handlers[handler_id] = (name, kind)

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        handler_id = self._get_handler_id(kwargs["tags"])
        span = self._create_span(handler_id)
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            json.dumps({"inputs": inputs, "kwargs": kwargs}, cls=CustomJsonEncode),
        )

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        handler_id = self._get_handler_id(kwargs["tags"])
        span = self._get_span(handler_id)
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            json.dumps({"outputs": outputs, "kwargs": kwargs}, cls=CustomJsonEncode),
        )
        span.end()

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""
        handler_id = self._get_handler_id(kwargs["tags"])
        span = self._create_span(handler_id)
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            json.dumps(
                {"input_str": input_str, "kwargs": kwargs}, cls=CustomJsonEncode
            ),
        )

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Run when tool ends running."""
        handler_id = self._get_handler_id(kwargs["tags"])
        span = self._get_span(handler_id)
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            json.dumps({"output": output, "kwargs": kwargs}, cls=CustomJsonEncode),
        )
        span.end()
