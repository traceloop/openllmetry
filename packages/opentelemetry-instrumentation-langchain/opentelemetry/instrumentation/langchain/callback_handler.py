"""Callback handler for Langchain instrumentation."""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import (
    BaseCallbackHandler,
)
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    SpanAttributes,
    TraceloopSpanKindValues,
)
from opentelemetry.context.context import Context
from opentelemetry.trace import SpanKind, set_span_in_context, Tracer
from opentelemetry.trace.span import Span
from opentelemetry._events import EventLogger

from opentelemetry import context as context_api
from opentelemetry.instrumentation.langchain.utils import (
    CallbackFilteredJSONEncoder,
    dont_throw,
    should_send_prompts,
)
from opentelemetry.metrics import Histogram
from opentelemetry.instrumentation.langchain.config import Config
from opentelemetry.instrumentation.langchain.events import (
    create_prompt_event,
    create_completion_event,
    create_chain_event,
    create_tool_event,
    _message_type_to_role,
)


@dataclass
class SpanHolder:
    span: Span
    token: Any
    context: Context
    children: list[UUID]
    workflow_name: str
    entity_name: str
    entity_path: str
    start_time: float = field(default_factory=time.time)
    request_model: Optional[str] = None
def _set_span_attribute(span: Span, name: str, value: Any) -> None:
    """Set a span attribute if the value is not None.
    
    Args:
        span: The span to set the attribute on
        name: The name of the attribute
        value: The value to set
    """
    if value is not None:
        span.set_attribute(name, value)


def _set_request_params(span: Span, kwargs: Dict[str, Any], span_holder: SpanHolder) -> None:
    """Set common request parameters on the span and update span holder model.
    
    Args:
        span: The span to set attributes on
        kwargs: The keyword arguments containing model and parameter information
        span_holder: The span holder to update with model information
    """
    for model_tag in ("model", "model_id", "model_name"):
        if (model := kwargs.get(model_tag)) is not None:
            span_holder.request_model = model
            break
        elif (model := (kwargs.get("invocation_params") or {}).get(model_tag)) is not None:
            span_holder.request_model = model
            break
    else:
        model = "unknown"
    span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model)
    # response is not available for LLM requests (as opposed to chat)
    span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, model)

    if "invocation_params" in kwargs:
        params = kwargs["invocation_params"].get("params") or kwargs["invocation_params"]
    else:
        params = kwargs
    _set_span_attribute(
        span,
        SpanAttributes.LLM_REQUEST_MAX_TOKENS,
        params.get("max_tokens") or params.get("max_new_tokens"),
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TEMPERATURE, params.get("temperature")
    )
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TOP_P, params.get("top_p"))


def _set_llm_request(
    span: Span,
    serialized: Dict[str, Any],
    prompts: List[str],
    kwargs: Any,
    span_holder: SpanHolder,
) -> None:
    """Set LLM request attributes on the span.
    
    Args:
        span: The span to set attributes on
        serialized: The serialized request data
        prompts: List of prompt strings
        kwargs: Additional keyword arguments
        span_holder: The span holder instance
    """
    _set_request_params(span, kwargs, span_holder)

    if should_send_prompts():
        for i, msg in enumerate(prompts):
            span.set_attribute(
                f"{SpanAttributes.LLM_PROMPTS}.{i}.role",
                "user",
            )
            span.set_attribute(
                f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                msg,
            )


def _set_chat_request(
    span: Span,
    serialized: Dict[str, Any],
    messages: List[List[BaseMessage]],
    kwargs: Any,
    span_holder: SpanHolder,
) -> None:
    """Set chat request attributes on the span.
    
    Args:
        span: The span to set attributes on
        serialized: The serialized request data
        messages: List of message lists, where each message is a BaseMessage
        kwargs: Additional keyword arguments
        span_holder: The span holder instance
    """
    _set_request_params(span, serialized.get("kwargs", {}), span_holder)

    if should_send_prompts():
        # Handle function definitions if present
        for i, function in enumerate(
            kwargs.get("invocation_params", {}).get("functions", [])
        ):
            prefix = f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}"

            _set_span_attribute(span, f"{prefix}.name", function.get("name"))
            _set_span_attribute(
                span, f"{prefix}.description", function.get("description")
            )
            _set_span_attribute(
                span, f"{prefix}.parameters", json.dumps(function.get("parameters"))
            )

        # Handle messages with proper role mapping and content serialization
        i = 0
        for message in messages:
            for msg in message:
                span.set_attribute(
                    f"{SpanAttributes.LLM_PROMPTS}.{i}.role",
                    _message_type_to_role(msg.type),
                )
                # Handle both string and structured content
                if isinstance(msg.content, str):
                    span.set_attribute(
                        f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                        msg.content,
                    )
                else:
                    span.set_attribute(
                        f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                        json.dumps(msg.content, cls=CallbackFilteredJSONEncoder),
                    )
                i += 1


class TraceloopCallbackHandler(BaseCallbackHandler):
    """Callback handler for Langchain that emits OpenTelemetry traces and events."""

    def __init__(
        self,
        tracer: Tracer,
        event_logger: Optional[EventLogger] = None,
        duration_histogram: Optional[Histogram] = None,
        token_histogram: Optional[Histogram] = None,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize the callback handler.
        
        Args:
            tracer: The OpenTelemetry tracer
            event_logger: Optional event logger for emitting events
            duration_histogram: Optional histogram for duration metrics
            token_histogram: Optional histogram for token usage metrics
            config: Optional configuration
        """
        super().__init__()
        self.tracer = tracer
        self.event_logger = event_logger
        self.duration_histogram = duration_histogram
        self.token_histogram = token_histogram
        self.config = config or Config()
        self._spans: Dict[UUID, SpanHolder] = {}

    def _emit_event(self, event: Dict[str, Any], span: Optional[Span] = None) -> None:
        """Emit an event if event logger is configured."""
        if self.event_logger and (self.config.capture_content or not event.get("attributes", {}).get(SpanAttributes.LLM_COMPLETION)):
            if span:
                span_ctx = span.get_span_context()
                event["trace_id"] = span_ctx.trace_id
                event["span_id"] = span_ctx.span_id
                event["trace_flags"] = span_ctx.trace_flags
            self.event_logger.emit(event)

    @staticmethod
    def _get_name_from_callback(
        serialized: Dict[str, Any],
        _tags: Optional[List[str]] = None,
        _metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Get the name to be used for the span. Based on heuristic. Can be extended."""
        if serialized and "kwargs" in serialized and serialized["kwargs"].get("name"):
            return serialized["kwargs"]["name"]
        if kwargs.get("name"):
            return kwargs["name"]
        if serialized.get("name"):
            return serialized["name"]
        if "id" in serialized:
            return serialized["id"][-1]
        return "unknown"

    @dont_throw
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain starts running."""
        if not self._should_handle_event():
            return

        name = self._get_name_from_callback(serialized, tags, metadata, **kwargs)
        kind = (
            TraceloopSpanKindValues.WORKFLOW
            if parent_run_id is None or parent_run_id not in self._spans
            else TraceloopSpanKindValues.CHAIN
        )

        workflow_name = ""
        entity_path = ""
        if kind == TraceloopSpanKindValues.WORKFLOW:
            workflow_name = name
        else:
            workflow_name = self.get_workflow_name(str(parent_run_id)) if parent_run_id else ""
            entity_path = self.get_entity_path(str(parent_run_id)) if parent_run_id else ""

        span = self._create_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            span_name=name,
            kind=kind,
            workflow_name=workflow_name,
            entity_name=serialized.get("id", [""])[0],
            entity_path=entity_path,
            metadata=metadata,
        )

        if should_send_prompts():
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                json.dumps(
                    {
                        "inputs": inputs,
                        "tags": tags,
                        "metadata": metadata,
                        "kwargs": kwargs,
                    },
                    cls=CallbackFilteredJSONEncoder,
                ),
            )

        if self.event_logger:
            self._emit_event(
                create_chain_event(
                    chain_type=serialized.get("id", [""])[0],
                    inputs=inputs,
                    model=self._spans[run_id].request_model if run_id in self._spans else None,
                ),
                span,
            )

    @dont_throw
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain ends running."""
        if not self._should_handle_event():
            return

        span_holder = self._spans[run_id]
        span = span_holder.span

        if should_send_prompts():
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                json.dumps(
                    {"outputs": outputs, "kwargs": kwargs},
                    cls=CallbackFilteredJSONEncoder,
                ),
            )

        if parent_run_id is None:
            context_api.attach(
                context_api.set_value(
                    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, False
                )
            )

        if self.duration_histogram:
            self.duration_histogram.record(
                time.time() - span_holder.start_time,
                {"workflow_name": span_holder.workflow_name},
            )

        self._end_span(span, run_id)

    @dont_throw
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when Chat Model starts running."""
        if not self._should_handle_event():
            return

        name = self._get_name_from_callback(serialized, tags, metadata, **kwargs)
        span = self._create_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            span_name=name,
            kind=LLMRequestTypeValues.CHAT,
            metadata=metadata,
        )

        _set_chat_request(span, serialized, messages, kwargs, self._spans[run_id])

        if self.event_logger:
            for message_list in messages:
                for message in message_list:
                    self._emit_event(
                        create_prompt_event(
                            content=message,
                            model=self._spans[run_id].request_model if run_id in self._spans else None,
                        ),
                        span,
                    )

    @dont_throw
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running."""
        if not self._should_handle_event():
            return

        name = self._get_name_from_callback(serialized, tags, metadata, **kwargs)
        span = self._create_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            span_name=name,
            kind=LLMRequestTypeValues.COMPLETION,
            metadata=metadata,
        )

        _set_llm_request(span, serialized, prompts, kwargs, self._spans[run_id])

        if self.event_logger:
            for prompt in prompts:
                self._emit_event(
                    create_prompt_event(
                        content=prompt,
                        role="user",
                        model=self._spans[run_id].request_model if run_id in self._spans else None,
                    ),
                    span,
                )

    @dont_throw
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ):
        """Handle LLM end event."""
        if not self._should_handle_event():
            return

        span = self._get_span(run_id)
        if span:
            if hasattr(response, "generations"):
                for i, generation in enumerate(response.generations):
                    for j, gen in enumerate(generation):
                        if self.config.use_legacy_attributes:
                            span.set_attribute(
                                f"{SpanAttributes.LLM_COMPLETIONS}.{i}.{j}.content",
                                gen.text,
                            )
                        
                        if self.event_logger:
                            self._emit_event(
                                create_completion_event(
                                    completion=gen.text,
                                    model=self._spans[run_id].request_model if run_id in self._spans else None,
                                    role="assistant",
                                    finish_reason=getattr(gen, "finish_reason", None),
                                ),
                                span,
                            )

            if hasattr(response, "llm_output") and response.llm_output:
                if "token_usage" in response.llm_output:
                    token_usage = response.llm_output["token_usage"]
                    if self.token_histogram:
                        self.token_histogram.record(
                            token_usage.get("total_tokens", 0),
                            {
                                "type": "total",
                                "model": self._spans[run_id].request_model if run_id in self._spans else "unknown",
                            },
                        )

            self._end_span(span, run_id)

    @dont_throw
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
        """Handle tool start event."""
        if not self._should_handle_event():
            return

        span = self._create_task_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=self._get_name_from_callback(serialized, tags, metadata, **kwargs),
            kind=TraceloopSpanKindValues.TOOL,
            workflow_name=self.get_workflow_name(str(parent_run_id)) if parent_run_id else "",
            entity_name=serialized.get("name", ""),
            entity_path=self.get_entity_path(str(parent_run_id)) if parent_run_id else "",
            metadata=metadata,
        )

        if self.event_logger:
            self._emit_event(
                create_tool_event(
                    tool_name=serialized.get("name", ""),
                    tool_input={"input": input_str} if input_str else inputs or {},
                    model=self._spans[run_id].request_model if run_id in self._spans else None,
                ),
                span,
            )

    @dont_throw
    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool end event."""
        if not self._should_handle_event():
            return

        span = self._get_span(run_id)
        if span and self.event_logger:
            self._emit_event(
                create_tool_event(
                    tool_name=span.get_attribute(SpanAttributes.LLM_TOOL_NAME),
                    tool_input={},  # Already captured in start event
                    tool_output={"output": output} if output else {},
                    model=self._spans[run_id].request_model if run_id in self._spans else None,
                ),
                span,
            )

        self._end_span(span, run_id)

    def _should_handle_event(self) -> bool:
        """Check if the event should be handled."""
        return not (
            context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)
            or context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
        )

    def _get_span(self, run_id: UUID) -> Span:
        """Get a span by its run ID."""
        return self._spans[run_id].span

    def _end_span(self, span: Span, run_id: UUID) -> None:
        """End a span and all its children."""
        for child_id in self._spans[run_id].children:
            child_span = self._spans[child_id].span
            if child_span.end_time is None:  # avoid warning on ended spans
                child_span.end()
        span.end()

    def _create_span(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        span_name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        workflow_name: str = "",
        entity_name: str = "",
        entity_path: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """Create a new span with optional parent and metadata."""
        if metadata is not None:
            current_association_properties = (
                context_api.get_value("association_properties") or {}
            )
            nonnull_metadata = dict([k, v] for k, v in metadata.items() if v is not None)
            context_api.attach(
                context_api.set_value(
                    "association_properties",
                    {**current_association_properties, **nonnull_metadata},
                )
            )

        if parent_run_id is not None and parent_run_id in self._spans:
            span = self.tracer.start_span(
                span_name,
                context=set_span_in_context(self._spans[parent_run_id].span),
                kind=kind,
            )
        else:
            span = self.tracer.start_span(span_name, kind=kind)

        span.set_attribute(SpanAttributes.TRACELOOP_WORKFLOW_NAME, workflow_name)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_PATH, entity_path)

        token = context_api.attach(
            context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True)
        )

        self._spans[run_id] = SpanHolder(
            span, token, None, [], workflow_name, entity_name, entity_path
        )

        if parent_run_id is not None and parent_run_id in self._spans:
            self._spans[parent_run_id].children.append(run_id)

        return span

    def _create_task_span(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        name: str,
        kind: TraceloopSpanKindValues,
        workflow_name: str,
        entity_name: str = "",
        entity_path: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> Span:
        span_name = f"{name}.{kind.value}"
        span = self._create_span(
            run_id,
            parent_run_id,
            span_name,
            workflow_name=workflow_name,
            entity_name=entity_name,
            entity_path=entity_path,
            metadata=metadata,
        )

        span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, kind.value)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)

        return span

    def _create_llm_span(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        name: str,
        request_type: LLMRequestTypeValues,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Span:
        workflow_name = self.get_workflow_name(parent_run_id)
        entity_path = self.get_entity_path(parent_run_id)

        span = self._create_span(
            run_id,
            parent_run_id,
            f"{name}.{request_type.value}",
            kind=SpanKind.CLIENT,
            workflow_name=workflow_name,
            entity_path=entity_path,
            metadata=metadata,
        )
        span.set_attribute(SpanAttributes.LLM_SYSTEM, "Langchain")
        span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, request_type.value)

        return span

    def get_parent_span(self, parent_run_id: Optional[str] = None):
        if parent_run_id is None:
            return None
        return self._spans[parent_run_id]

    def get_workflow_name(self, run_id: str) -> str:
        """Get the workflow name for a given run ID."""
        return self._spans[UUID(run_id)].workflow_name if UUID(run_id) in self._spans else ""

    def get_entity_path(self, run_id: str) -> str:
        """Get the entity path for a given run ID."""
        return self._spans[UUID(run_id)].entity_path if UUID(run_id) in self._spans else ""
