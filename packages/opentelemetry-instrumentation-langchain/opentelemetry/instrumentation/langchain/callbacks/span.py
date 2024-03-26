"""Callback Handler that creates and destroys span."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.utils import print_text
from opentelemetry import context, propagators
from opentelemetry.semconv.ai import SpanAttributes, TraceloopSpanKindValues
from opentelemetry.trace import Status, StatusCode, use_span, get_current_span, set_span_in_context

if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish


class SpanCallbackHandler(BaseCallbackHandler):
    """Callback Handler that creates and destroys span."""

    def __init__(self, tracer: "Tracer", name: Optional[str] = None, kind: Optional[str] = None,  color: Optional[str] = None) -> None:
        """Initialize callback handler."""
        super().__init__()
        self.color = color
        self._span_dict: Dict[str, "SPAN"] = {}
        self._tracer = tracer
        self._name = name
        self._kind = kind

    def _create_span(self) -> None:
        parent = get_current_span()
        span = self._tracer.start_span(self._name)
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            self._kind,
        )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, self._name)
        self._span_dict[self._name] = span
        
        if parent.is_recording():
          set_span_in_context(span)
        activation = use_span(span, end_on_exit=True)
        activation.__enter__()
    def _error_span(self, error) -> None:
      current_span = self._span_dict.get(self._name, None)
      if current_span:
          current_span.set_status(Status(StatusCode.ERROR))
          current_span.record_exception(error)
          current_span.end()
    def _end_span(self) -> None:
      current_span = self._span_dict.get(self._name, None)
      if current_span:
        current_span.end()
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        if not self._name:
          class_name = serialized.get("name", serialized.get("id", ["unknown"])[-1])
          self._name = f"langchain.task.{class_name}"

        if not self._kind:
          if serialized.get('name') in ('SequentialChain'):
            self._kind = TraceloopSpanKindValues.WORKFLOW.value
          else:
            self._kind = TraceloopSpanKindValues.TASK.value
        self._create_span()
        
        print(f"\n\n\033[1m> Entering new {self._name} chain...\033[0m")  # noqa: T201
        

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        self._error_span(error)        
        print("\n\033[1m> Errored out chain.\033[0m")  # noqa: T201

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        self._end_span()
        print("\n\033[1m> Finished chain.\033[0m")  # noqa: T201

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action."""
        print("On agent action")

    def on_tool_end(
        self,
        output: Any,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        output = str(output)
        if observation_prefix is not None:
            print_text(f"\n{observation_prefix}")
        print_text(output, color=color or self.color)
        if llm_prefix is not None:
            print_text(f"\n{llm_prefix}")
        print("On tool end")

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        """Run on arbitrary text."""
        print("On text")
        print_text(text, color=color or self.color, end=end)

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        print("On agent finish")
        print_text(finish.log, color=color or self.color, end="\n")