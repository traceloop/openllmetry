"""Callback Handler that creates and destroys span."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.utils import print_text
from opentelemetry import context, propagators
from opentelemetry.trace.span import Span
from opentelemetry.semconv.ai import SpanAttributes, TraceloopSpanKindValues
from opentelemetry.trace import Status, StatusCode, use_span, get_current_span, set_span_in_context
from opentelemetry import context as context_api
from opentelemetry.trace.propagation import _SPAN_KEY

if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish


class SpanCallbackHandler(BaseCallbackHandler):
    """Callback Handler that creates and destroys span."""

    def __init__(self, tracer: Tracer, span_name: Optional[str] = None, kind: Optional[str] = None,  color: Optional[str] = None) -> None:
        """Initialize callback handler."""
        super().__init__()
        self.color = color
        self._span_dict: Dict[str, Span] = {}
        self._tracer = tracer
        self._span_name = span_name
        self._name = None
        self._kind = kind

    def _create_span(self, key, name, kind) -> None:
        span = self._tracer.start_span(name)
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            kind,
        )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name)
        curr_context = context_api.set_value(_SPAN_KEY, span)
        if kind == TraceloopSpanKindValues.WORKFLOW.value:
            curr_context = context_api.set_value("workflow_name", name, context=curr_context)
            # span.set_attribute("workflow_name", name)
        elif context_api.get_value("workflow_name"):
            curr_context = context_api.set_value("workflow_name", context_api.get_value("workflow_name"), context=curr_context)
            # span.set_attribute("workflow_name", context_api.get_value("workflow_name"))
        token = context_api.attach(curr_context)
        self._span_dict[key] = {"span": span, "token": token}
        print(name, kind, self._span_dict)
        # if self._kind == TraceloopSpanKindValues.WORKFLOW.value:
        #    context_api.attach(context_api.set_value("workflow_name", self._name))
        # print(token)
        
        # if parent.is_recording():
        #   set_span_in_context(span)
        use_span(span)
        # activation.__enter__()
    def _error_span(self, error, key) -> None:
        span_dict_entry = self._span_dict.get(key, None)
        span = span_dict_entry.get("span", None)
        token = span_dict_entry.get("token", None)
        if token:
            context_api.detach(token)
        if isinstance(span, Span)  and span.is_recording():
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(error)
            span.end()
    def _end_span(self, key) -> None:
        span_dict_entry = self._span_dict.get(key, None)
        span = span_dict_entry.get("span", None)
        token = span_dict_entry.get("token", None)
        if token:
            print('token', token)
            context_api.detach(token)
        if isinstance(span, Span)  and span.is_recording():
            print('ending', key)
            span.end()

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        print(f"\n\n\033[1m> Entering new {self._name} chain...\033[0m")  # noqa: T201
        class_name = serialized.get("name", serialized.get("id", ["unknown"])[-1])
        if self._span_name == "langchain.workflow":
           self._create_span(self._span_name, self._span_name, TraceloopSpanKindValues.WORKFLOW.value)
        self._name = f"langchain.task.{class_name}"
        self._kind = TraceloopSpanKindValues.TASK.value
        self._create_span(kwargs.get('run_id', self._name), self._name, self._kind)
        print(f"\n\n\033[1m> Done Entering new {self._name} chain...\033[0m")  # noqa: T201
        

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        print("\n\033[1m> Errored out chain.\033[0m")  # noqa: T201
        if self._span_name == "langchain.workflow":
            self._end_span(error, self._span_name)
        self._error_span(error, kwargs.get('run_id', self._name))
        print("\n\033[1m> Done Errored out chain.\033[0m")  # noqa: T201
        

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        print("\n\033[1m> Finished chain.\033[0m")  # noqa: T201
        self._end_span(kwargs.get('run_id', self._name))
        if self._span_name == "langchain.workflow":
            self._end_span(self._span_name)
        print("\n\033[1m> Done Finished chain.\033[0m")  # noqa: T201
        

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