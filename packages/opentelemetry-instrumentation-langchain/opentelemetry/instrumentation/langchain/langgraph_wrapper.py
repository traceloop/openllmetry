"""Tool invocation wrapper for OpenTelemetry Langchain instrumentation"""

from typing import Any, Callable
from opentelemetry.trace.propagation import set_span_in_context
from opentelemetry import context as context_api
from opentelemetry.instrumentation.langchain.callback_handler import TraceloopCallbackHandler


class LanggraphWrapper:
    """Wrapper for tool invocation that sets the current span in context."""

    def __init__(self, callback_manager: TraceloopCallbackHandler):
        self._callback_manager = callback_manager

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:        
        print("instance", args)
        print("kwargs", kwargs)
        # if len(args) >= 2 and isinstance(args[1], dict):
        #     metadata = args[1].get('metadata', {})
        #     checkpoint_ns = metadata.get('langgraph_checkpoint_ns')

        # if checkpoint_ns:
        #     span_holder = self._callback_manager._get_span_by_checkpoint_ns(checkpoint_ns)
        #     set_span_in_context(span_holder.span)
        
        return wrapped(*args, **kwargs) 