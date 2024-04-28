"""Callback Handler that creates and destroys span."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from langchain_core.callbacks.base import BaseCallbackHandler, AsyncCallbackHandler
from langchain_core.utils import print_text
from opentelemetry import context, propagators
from opentelemetry.trace.span import Span
from opentelemetry.semconv.ai import SpanAttributes, TraceloopSpanKindValues
from opentelemetry.trace import Status, StatusCode, use_span, get_current_span, set_span_in_context
from opentelemetry import context as context_api
from opentelemetry.trace.propagation import _SPAN_KEY

if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish

PARENT_SPAN_MAPPING = {
    "SequentialChain" : {
        "span_name": "langchain.workflow",
        "kind": TraceloopSpanKindValues.WORKFLOW.value,
    },
    "Agent": {
        "span_name": "langchain.agent",
        "kind": TraceloopSpanKindValues.AGENT.value,
    },
    "AgentExecutor": {
        "span_name": "langchain.agent",
        "kind": TraceloopSpanKindValues.AGENT.value,
    },
}

class SyncSpanCallbackHandler(BaseCallbackHandler):
    """Callback Handler that creates and destroys span."""

    def __init__(self, tracer: Tracer, span_name: Optional[str] = None, kind: Optional[str] = None,  color: Optional[str] = None) -> None:
        """Initialize callback handler."""
        # super().__init__()
        self.color = color
        self._span_dict: Dict[str, Span] = {}
        self._tracer = tracer
        self._span_name = span_name
        self._name = None
        self._kind = kind

    def _create_span(self, key, name, kind) -> None:

        # Create a Span
        span = self._tracer.start_span(name)
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            kind,
        )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name)

        # Create a context
        curr_context = context_api.set_value(_SPAN_KEY, span)
        if kind in (TraceloopSpanKindValues.WORKFLOW.value, TraceloopSpanKindValues.AGENT.value):
            curr_context = context_api.set_value("workflow_name", name, context=curr_context)
        elif context_api.get_value("workflow_name"):
            curr_context = context_api.set_value("workflow_name", context_api.get_value("workflow_name"), context=curr_context)
        
        print("Create span ", span)
        # Attach context to its parent
        token = context_api.attach(curr_context)

        # Update span_dict
        self._span_dict[key] = {"span": span, "token": token}

        # Activate the Span
        use_span(span)

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
        class_name = serialized.get("name", serialized.get("id", ["unknown"])[-1])
        print('class name', class_name)
        if class_name in PARENT_SPAN_MAPPING:
            self._span_name = PARENT_SPAN_MAPPING[class_name]["span_name"]
            self._create_span(self._span_name, self._span_name, PARENT_SPAN_MAPPING[class_name]["kind"])
        print(f"\n\n\033[1m> Entering new {class_name} chain...\033[0m")  # noqa: T201
        # # We just spawn a AgentExecutor and
        # if class_name not in ("AgentExecutor"):
        self._name = f"langchain.task.{class_name}"
        self._kind = TraceloopSpanKindValues.TASK.value
        self._create_span(kwargs.get('run_id', self._name), self._name, self._kind)
        print(f"\n\n\033[1m> Done Entering new {class_name} chain...\033[0m")  # noqa: T201
        

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        print(f"\n\033[1m> Errored out {self._name} chain.\033[0m")  # noqa: T201
        if self._span_name:
            self._end_span(error, self._span_name)
        self._error_span(error, kwargs.get('run_id', self._name))
        print(f"\n\033[1m> Done Errored out {self._name} chain.\033[0m")  # noqa: T201
        

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        print("\n\033[1m> Finished chain.\033[0m")  # noqa: T201
        self._end_span(kwargs.get('run_id', self._name))
        if self._span_name:
            self._end_span(self._span_name)
        print("\n\033[1m> Done Finished chain.\033[0m")  # noqa: T201
        
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        class_name = serialized.get("name", serialized.get("id", ["unknown"])[-1])
        print(f"\n\n\033[1m> Entering new {class_name} tool...\033[0m")  # noqa: T201
        print('class name', class_name)
        self._span_name = "langchain.tool"
        self._name = f"{self._span_name}.{class_name}"
        self._kind = TraceloopSpanKindValues.TOOL.value
        self._create_span(kwargs.get('run_id', self._name), self._name, self._kind)
        print(f"\n\n\033[1m> Done Entering new {class_name} tool...\033[0m")  # noqa: T201

    def on_tool_end(
        self,
        output: Any,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        print("\n\033[1m> Finished tool.\033[0m")  # noqa: T201
        self._end_span(kwargs.get('run_id', self._name))
        print("\n\033[1m> Done Finished tool.\033[0m")  # noqa: T201

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        print("\n\033[1m> Errored out tool.\033[0m")  # noqa: T201
        self._error_span(error, kwargs.get('run_id', self._name))
        print("\n\033[1m> Done Errored out tool.\033[0m")  # noqa: T201

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

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action."""
        print("On agent action")

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        print("On agent finish")
        print_text(finish.log, color=color or self.color, end="\n")


class AsyncSpanCallbackHandler(AsyncCallbackHandler):
    """Async Callback Handler that creates and destroys span."""

    def __init__(self, tracer: Tracer, span_name: Optional[str] = None, kind: Optional[str] = None,  color: Optional[str] = None) -> None:
        """Initialize callback handler."""
        # super().__init__()
        self.color = color
        self._span_dict: Dict[str, Span] = {}
        self._tracer = tracer
        self._span_name = span_name
        self._name = None
        self._kind = kind

    def _create_span(self, key, name, kind) -> None:

        # Create a Span
        span = self._tracer.start_span(name)
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            kind,
        )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name)

        # Create a context
        curr_context = context_api.set_value(_SPAN_KEY, span)
        if kind in (TraceloopSpanKindValues.WORKFLOW.value, TraceloopSpanKindValues.AGENT.value):
            curr_context = context_api.set_value("workflow_name", name, context=curr_context)
        elif context_api.get_value("workflow_name"):
            curr_context = context_api.set_value("workflow_name", context_api.get_value("workflow_name"), context=curr_context)
        
        print("Create span ", span)
        # Attach context to its parent
        token = context_api.attach(curr_context)

        # Update span_dict
        self._span_dict[key] = {"span": span, "token": token}

        # Activate the Span
        use_span(span)

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

    async def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized.get("name", serialized.get("id", ["unknown"])[-1])
        print('class name', class_name)
        if class_name in PARENT_SPAN_MAPPING:
            self._span_name = PARENT_SPAN_MAPPING[class_name]["span_name"]
            await self._create_span(self._span_name, self._span_name, PARENT_SPAN_MAPPING[class_name]["kind"])
        print(f"\n\n\033[1m> Entering new {class_name} chain...\033[0m")  # noqa: T201
        # # We just spawn a AgentExecutor and
        # if class_name not in ("AgentExecutor"):
        self._name = f"langchain.task.{class_name}"
        self._kind = TraceloopSpanKindValues.TASK.value
        await self._create_span(kwargs.get('run_id', self._name), self._name, self._kind)
        print(f"\n\n\033[1m> Done Entering new {class_name} chain...\033[0m")  # noqa: T201
        

    async def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        print(f"\n\033[1m> Errored out {self._name} chain.\033[0m")  # noqa: T201
        if self._span_name:
            self._end_span(error, self._span_name)
        await self._error_span(error, kwargs.get('run_id', self._name))
        print(f"\n\033[1m> Done Errored out {self._name} chain.\033[0m")  # noqa: T201
        

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        print("\n\033[1m> Finished chain.\033[0m")  # noqa: T201
        await self._end_span(kwargs.get('run_id', self._name))
        if self._span_name:
            await self._end_span(self._span_name)
        print("\n\033[1m> Done Finished chain.\033[0m")  # noqa: T201
        
    async def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        class_name = serialized.get("name", serialized.get("id", ["unknown"])[-1])
        print(f"\n\n\033[1m> Entering new {class_name} tool...\033[0m")  # noqa: T201
        print('class name', class_name)
        self._span_name = "langchain.tool"
        self._name = f"{self._span_name}.{class_name}"
        self._kind = TraceloopSpanKindValues.TOOL.value
        await self._create_span(kwargs.get('run_id', self._name), self._name, self._kind)
        print(f"\n\n\033[1m> Done Entering new {class_name} tool...\033[0m")  # noqa: T201

    async def on_tool_end(
        self,
        output: Any,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        print("\n\033[1m> Finished tool.\033[0m")  # noqa: T201
        await self._end_span(kwargs.get('run_id', self._name))
        print("\n\033[1m> Done Finished tool.\033[0m")  # noqa: T201

    async def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        print("\n\033[1m> Errored out tool.\033[0m")  # noqa: T201
        await self._error_span(error, kwargs.get('run_id', self._name))
        print("\n\033[1m> Done Errored out tool.\033[0m")  # noqa: T201

    async def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        """Run on arbitrary text."""
        print("On text")
        print_text(text, color=color or self.color, end=end)

    async def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action."""
        print("On agent action")

    async def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        print("On agent finish")
        print_text(finish.log, color=color or self.color, end="\n")