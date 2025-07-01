import json
import time
from typing import Any, Dict, List, Optional, Type, Union
from uuid import UUID

from langchain_core.callbacks import (
    BaseCallbackHandler,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    Generation,
    GenerationChunk,
    LLMResult,
)
from opentelemetry import context as context_api
from opentelemetry.instrumentation.langchain.event_emitter import emit_event
from opentelemetry.instrumentation.langchain.event_models import (
    ChoiceEvent,
    MessageEvent,
    ToolCall,
)
from opentelemetry.instrumentation.langchain.span_utils import (
    SpanHolder,
    _set_span_attribute,
    set_chat_request,
    set_chat_response,
    set_chat_response_usage,
    set_llm_request,
    set_request_params,
)
from opentelemetry.instrumentation.langchain.utils import (
    CallbackFilteredJSONEncoder,
    dont_throw,
    should_emit_events,
    should_send_prompts,
)
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.metrics import Histogram
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_RESPONSE_ID,
)
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    SpanAttributes,
    TraceloopSpanKindValues,
)
from opentelemetry.trace import SpanKind, Tracer, set_span_in_context
from opentelemetry.trace.span import Span
from opentelemetry.trace.status import Status, StatusCode


def _message_type_to_role(message_type: str) -> str:
    if message_type == "human":
        return "user"
    elif message_type == "system":
        return "system"
    elif message_type == "ai":
        return "assistant"
    elif message_type == "tool":
        return "tool"
    else:
        return "unknown"


def _sanitize_metadata_value(value: Any) -> Any:
    """Convert metadata values to OpenTelemetry-compatible types."""
    if value is None:
        return None
    if isinstance(value, (bool, str, bytes, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [str(_sanitize_metadata_value(v)) for v in value]
    # Convert other types to strings
    return str(value)


def valid_role(role: str) -> bool:
    return role in ["user", "assistant", "system", "tool"]


def get_message_role(message: Type[BaseMessage]) -> str:
    if isinstance(message, (SystemMessage, SystemMessageChunk)):
        return "system"
    elif isinstance(message, (HumanMessage, HumanMessageChunk)):
        return "user"
    elif isinstance(message, (AIMessage, AIMessageChunk)):
        return "assistant"
    elif isinstance(message, (ToolMessage, ToolMessageChunk)):
        return "tool"
    else:
        return "unknown"


def _extract_tool_call_data(
    tool_calls: Optional[List[dict[str, Any]]],
) -> Union[List[ToolCall], None]:
    if tool_calls is None:
        return tool_calls

    response = []

    for tool_call in tool_calls:
        tool_call_function = {"name": tool_call.get("name", "")}

        if tool_call.get("arguments"):
            tool_call_function["arguments"] = tool_call["arguments"]
        elif tool_call.get("args"):
            tool_call_function["arguments"] = tool_call["args"]
        response.append(
            ToolCall(
                id=tool_call.get("id", ""),
                function=tool_call_function,
                type="function",
            )
        )

    return response


class TraceloopCallbackHandler(BaseCallbackHandler):
    def __init__(
        self, tracer: Tracer, duration_histogram: Histogram, token_histogram: Histogram
    ) -> None:
        super().__init__()
        self.tracer = tracer
        self.duration_histogram = duration_histogram
        self.token_histogram = token_histogram
        self.spans: dict[UUID, SpanHolder] = {}
        self.run_inline = True

    @staticmethod
    def _get_name_from_callback(
        serialized: dict[str, Any],
        _tags: Optional[list[str]] = None,
        _metadata: Optional[dict[str, Any]] = None,
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

    def _get_span(self, run_id: UUID) -> Span:
        return self.spans[run_id].span

    def _end_span(self, span: Span, run_id: UUID) -> None:
        for child_id in self.spans[run_id].children:
            child_span = self.spans[child_id].span
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
        metadata: Optional[dict[str, Any]] = None,
    ) -> Span:
        if metadata is not None:
            current_association_properties = (
                context_api.get_value("association_properties") or {}
            )
            # Sanitize metadata values to ensure they're compatible with OpenTelemetry
            sanitized_metadata = {
                k: _sanitize_metadata_value(v)
                for k, v in metadata.items()
                if v is not None
            }
            context_api.attach(
                context_api.set_value(
                    "association_properties",
                    {**current_association_properties, **sanitized_metadata},
                )
            )

        if parent_run_id is not None and parent_run_id in self.spans:
            span = self.tracer.start_span(
                span_name,
                context=set_span_in_context(self.spans[parent_run_id].span),
                kind=kind,
            )
        else:
            span = self.tracer.start_span(span_name, kind=kind)

        _set_span_attribute(span, SpanAttributes.TRACELOOP_WORKFLOW_NAME, workflow_name)
        _set_span_attribute(span, SpanAttributes.TRACELOOP_ENTITY_PATH, entity_path)

        token = context_api.attach(
            context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True)
        )

        self.spans[run_id] = SpanHolder(
            span, token, None, [], workflow_name, entity_name, entity_path
        )

        if parent_run_id is not None and parent_run_id in self.spans:
            self.spans[parent_run_id].children.append(run_id)

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

        _set_span_attribute(span, SpanAttributes.TRACELOOP_SPAN_KIND, kind.value)
        _set_span_attribute(span, SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)

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
        _set_span_attribute(span, SpanAttributes.LLM_SYSTEM, "Langchain")
        _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TYPE, request_type.value)

        return span

    @dont_throw
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
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        workflow_name = ""
        entity_path = ""

        name = self._get_name_from_callback(serialized, **kwargs)
        kind = (
            TraceloopSpanKindValues.WORKFLOW
            if parent_run_id is None or parent_run_id not in self.spans
            else TraceloopSpanKindValues.TASK
        )

        if kind == TraceloopSpanKindValues.WORKFLOW:
            workflow_name = name
        else:
            workflow_name = self.get_workflow_name(parent_run_id)
            entity_path = self.get_entity_path(parent_run_id)

        span = self._create_task_span(
            run_id,
            parent_run_id,
            name,
            kind,
            workflow_name,
            name,
            entity_path,
            metadata,
        )
        if not should_emit_events() and should_send_prompts():
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

        # The start_time is now automatically set when creating the SpanHolder

    @dont_throw
    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain ends running."""
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        span_holder = self.spans[run_id]
        span = span_holder.span
        if not should_emit_events() and should_send_prompts():
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                json.dumps(
                    {"outputs": outputs, "kwargs": kwargs},
                    cls=CallbackFilteredJSONEncoder,
                ),
            )

        self._end_span(span, run_id)
        if parent_run_id is None:
            context_api.attach(
                context_api.set_value(
                    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, False
                )
            )

    @dont_throw
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Chat Model starts running."""
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        name = self._get_name_from_callback(serialized, kwargs=kwargs)
        span = self._create_llm_span(
            run_id, parent_run_id, name, LLMRequestTypeValues.CHAT, metadata=metadata
        )
        set_request_params(span, kwargs, self.spans[run_id])
        if should_emit_events():
            self._emit_chat_input_events(messages)
        else:
            set_chat_request(span, serialized, messages, kwargs, self.spans[run_id])

    @dont_throw
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Chat Model starts running."""
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        name = self._get_name_from_callback(serialized, kwargs=kwargs)
        span = self._create_llm_span(
            run_id, parent_run_id, name, LLMRequestTypeValues.COMPLETION
        )
        set_request_params(span, kwargs, self.spans[run_id])
        if should_emit_events():
            for prompt in prompts:
                emit_event(MessageEvent(content=prompt, role="user"))
        else:
            set_llm_request(span, serialized, prompts, kwargs, self.spans[run_id])

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        span = self._get_span(run_id)

        model_name = None
        if response.llm_output is not None:
            model_name = response.llm_output.get(
                "model_name"
            ) or response.llm_output.get("model_id")
            if model_name is not None:
                _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, model_name)

                if self.spans[run_id].request_model is None:
                    _set_span_attribute(
                        span, SpanAttributes.LLM_REQUEST_MODEL, model_name
                    )
            id = response.llm_output.get("id")
            if id is not None and id != "":
                _set_span_attribute(span, GEN_AI_RESPONSE_ID, id)

        token_usage = (response.llm_output or {}).get("token_usage") or (
            response.llm_output or {}
        ).get("usage")
        if token_usage is not None:
            prompt_tokens = (
                token_usage.get("prompt_tokens")
                or token_usage.get("input_token_count")
                or token_usage.get("input_tokens")
            )
            completion_tokens = (
                token_usage.get("completion_tokens")
                or token_usage.get("generated_token_count")
                or token_usage.get("output_tokens")
            )
            total_tokens = token_usage.get("total_tokens") or (
                prompt_tokens + completion_tokens
            )

            _set_span_attribute(
                span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens
            )
            _set_span_attribute(
                span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens
            )
            _set_span_attribute(
                span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens
            )

            # Record token usage metrics
            if prompt_tokens > 0:
                self.token_histogram.record(
                    prompt_tokens,
                    attributes={
                        SpanAttributes.LLM_SYSTEM: "Langchain",
                        SpanAttributes.LLM_TOKEN_TYPE: "input",
                        SpanAttributes.LLM_RESPONSE_MODEL: model_name or "unknown",
                    },
                )

            if completion_tokens > 0:
                self.token_histogram.record(
                    completion_tokens,
                    attributes={
                        SpanAttributes.LLM_SYSTEM: "Langchain",
                        SpanAttributes.LLM_TOKEN_TYPE: "output",
                        SpanAttributes.LLM_RESPONSE_MODEL: model_name or "unknown",
                    },
                )
        set_chat_response_usage(span, response)
        if should_emit_events():
            self._emit_llm_end_events(response)
        else:
            set_chat_response(span, response)
        self._end_span(span, run_id)

        # Record duration
        duration = time.time() - self.spans[run_id].start_time
        self.duration_histogram.record(
            duration,
            attributes={
                SpanAttributes.LLM_SYSTEM: "Langchain",
                SpanAttributes.LLM_RESPONSE_MODEL: model_name or "unknown",
            },
        )

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
        """Run when tool starts running."""
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        name = self._get_name_from_callback(serialized, kwargs=kwargs)
        workflow_name = self.get_workflow_name(parent_run_id)
        entity_path = self.get_entity_path(parent_run_id)

        span = self._create_task_span(
            run_id,
            parent_run_id,
            name,
            TraceloopSpanKindValues.TOOL,
            workflow_name,
            name,
            entity_path,
        )
        if not should_emit_events() and should_send_prompts():
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
                    cls=CallbackFilteredJSONEncoder,
                ),
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
        """Run when tool ends running."""
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        span = self._get_span(run_id)
        if not should_emit_events() and should_send_prompts():
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                json.dumps(
                    {"output": output, "kwargs": kwargs},
                    cls=CallbackFilteredJSONEncoder,
                ),
            )
        self._end_span(span, run_id)

    def get_parent_span(self, parent_run_id: Optional[str] = None):
        if parent_run_id is None:
            return None
        return self.spans[parent_run_id]

    def get_workflow_name(self, parent_run_id: str):
        parent_span = self.get_parent_span(parent_run_id)

        if parent_span is None:
            return ""

        return parent_span.workflow_name

    def get_entity_path(self, parent_run_id: str):
        parent_span = self.get_parent_span(parent_run_id)

        if parent_span is None:
            return ""
        elif (
            parent_span.entity_path == ""
            and parent_span.entity_name == parent_span.workflow_name
        ):
            return ""
        elif parent_span.entity_path == "":
            return f"{parent_span.entity_name}"
        else:
            return f"{parent_span.entity_path}.{parent_span.entity_name}"

    def _handle_error(
        self,
        error: BaseException,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Common error handling logic for all components."""
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        span = self._get_span(run_id)
        span.set_status(Status(StatusCode.ERROR))
        span.record_exception(error)
        self._end_span(span, run_id)

    @dont_throw
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors."""
        self._handle_error(error, run_id, parent_run_id, **kwargs)

    @dont_throw
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors."""
        self._handle_error(error, run_id, parent_run_id, **kwargs)

    @dont_throw
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool errors."""
        self._handle_error(error, run_id, parent_run_id, **kwargs)

    @dont_throw
    def on_agent_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when agent errors."""
        self._handle_error(error, run_id, parent_run_id, **kwargs)

    @dont_throw
    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when retriever errors."""
        self._handle_error(error, run_id, parent_run_id, **kwargs)

    def _emit_chat_input_events(self, messages):
        for message_list in messages:
            for message in message_list:
                if hasattr(message, "tool_calls") and message.tool_calls:
                    tool_calls = _extract_tool_call_data(message.tool_calls)
                else:
                    tool_calls = None
                emit_event(
                    MessageEvent(
                        content=message.content,
                        role=get_message_role(message),
                        tool_calls=tool_calls,
                    )
                )

    def _emit_llm_end_events(self, response):
        for generation_list in response.generations:
            for i, generation in enumerate(generation_list):
                self._emit_generation_choice_event(index=i, generation=generation)

    def _emit_generation_choice_event(
        self,
        index: int,
        generation: Union[
            ChatGeneration, ChatGenerationChunk, Generation, GenerationChunk
        ],
    ):
        if isinstance(generation, (ChatGeneration, ChatGenerationChunk)):
            # Get finish reason
            if hasattr(generation, "generation_info") and generation.generation_info:
                finish_reason = generation.generation_info.get(
                    "finish_reason", "unknown"
                )
            else:
                finish_reason = "unknown"

            # Get tool calls
            if (
                hasattr(generation.message, "tool_calls")
                and generation.message.tool_calls
            ):
                tool_calls = _extract_tool_call_data(generation.message.tool_calls)
            elif hasattr(
                generation.message, "additional_kwargs"
            ) and generation.message.additional_kwargs.get("function_call"):
                tool_calls = _extract_tool_call_data(
                    [generation.message.additional_kwargs.get("function_call")]
                )
            else:
                tool_calls = None

            # Emit the event
            if hasattr(generation, "text") and generation.text != "":
                emit_event(
                    ChoiceEvent(
                        index=index,
                        message={"content": generation.text, "role": "assistant"},
                        finish_reason=finish_reason,
                        tool_calls=tool_calls,
                    )
                )
            else:
                emit_event(
                    ChoiceEvent(
                        index=index,
                        message={
                            "content": generation.message.content,
                            "role": "assistant",
                        },
                        finish_reason=finish_reason,
                        tool_calls=tool_calls,
                    )
                )
        elif isinstance(generation, (Generation, GenerationChunk)):
            # Get finish reason
            if hasattr(generation, "generation_info") and generation.generation_info:
                finish_reason = generation.generation_info.get(
                    "finish_reason", "unknown"
                )
            else:
                finish_reason = "unknown"

            # Emit the event
            emit_event(
                ChoiceEvent(
                    index=index,
                    message={"content": generation.text, "role": "assistant"},
                    finish_reason=finish_reason,
                )
            )
