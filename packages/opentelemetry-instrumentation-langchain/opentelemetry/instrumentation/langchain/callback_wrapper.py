import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import (
    BaseCallbackHandler,
    BaseCallbackManager,
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

from opentelemetry import context as context_api
from opentelemetry.instrumentation.langchain.utils import (
    _with_tracer_wrapper,
    dont_throw,
    should_send_prompts,
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
    workflow_name: str
    entity_name: str
    entity_path: str


@dont_throw
def _add_callback(
    tracer, callbacks: Union[List[BaseCallbackHandler], BaseCallbackManager]
):
    cb = SyncSpanCallbackHandler(tracer)
    if isinstance(callbacks, BaseCallbackManager):
        for c in callbacks.handlers:
            if isinstance(c, SyncSpanCallbackHandler):
                cb = c
                break
        else:
            callbacks.add_handler(cb)
    elif isinstance(callbacks, list):
        for c in callbacks:
            if isinstance(c, SyncSpanCallbackHandler):
                cb = c
                break
        else:
            callbacks.append(cb)


@_with_tracer_wrapper
def callback_wrapper(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Hook into the invoke function, config is part of args, 2nd place.
    sources:
    https://python.langchain.com/v0.2/docs/how_to/callbacks_attach/
    https://python.langchain.com/v0.2/docs/how_to/callbacks_runtime/
    """
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    if len(args) > 1:
        # args[1] is config which (may) contain the callbacks setting
        callbacks = args[1].get("callbacks", [])
    elif kwargs.get("config"):
        callbacks = kwargs.get("config", {}).get("callbacks", [])
    else:
        callbacks = []

    _add_callback(tracer, callbacks)

    if len(args) > 1:
        args[1]["callbacks"] = callbacks
    elif kwargs.get("config"):
        kwargs["config"]["callbacks"] = callbacks
    else:
        kwargs["config"] = {"callbacks": callbacks}

    return wrapped(*args, **kwargs)


def _message_type_to_role(message_type: str) -> str:
    if message_type == "human":
        return "user"
    elif message_type == "system":
        return "system"
    elif message_type == "ai":
        return "assistant"
    else:
        return "unknown"


def _set_span_attribute(span, name, value):
    if value is not None:
        span.set_attribute(name, value)


def _set_request_params(span, kwargs):
    for model_tag in ("model", "model_id", "model_name"):
        if (model := kwargs.get(model_tag)) is not None:
            break
        elif (
            model := (kwargs.get("invocation_params") or {}).get(model_tag)
        ) is not None:
            break
    else:
        model = "unknown"
    span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model)
    # response is not available for LLM requests (as opposed to chat)
    span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, model)

    if "invocation_params" in kwargs:
        params = (
            kwargs["invocation_params"].get("params") or kwargs["invocation_params"]
        )
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
    serialized: dict[str, Any],
    prompts: list[str],
    kwargs: Any,
) -> None:
    _set_request_params(span, kwargs)

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
    serialized: dict[str, Any],
    messages: list[list[BaseMessage]],
    kwargs: Any,
) -> None:
    _set_request_params(span, serialized.get("kwargs", {}))

    if should_send_prompts():
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

        i = 0
        for message in messages:
            for msg in message:
                span.set_attribute(
                    f"{SpanAttributes.LLM_PROMPTS}.{i}.role",
                    _message_type_to_role(msg.type),
                )
                # if msg.content is string
                if isinstance(msg.content, str):
                    span.set_attribute(
                        f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                        msg.content,
                    )
                else:
                    span.set_attribute(
                        f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                        json.dumps(msg.content, cls=CustomJsonEncode),
                    )
                i += 1


def _set_chat_response(span: Span, response: LLMResult) -> None:
    if not should_send_prompts():
        return

    i = 0
    for generations in response.generations:
        for generation in generations:
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{i}"
            if hasattr(generation, "text") and generation.text != "":
                span.set_attribute(
                    f"{prefix}.content",
                    generation.text,
                )
                span.set_attribute(f"{prefix}.role", "assistant")
            else:
                span.set_attribute(
                    f"{prefix}.role",
                    _message_type_to_role(generation.type),
                )
                if generation.message.content is str:
                    span.set_attribute(
                        f"{prefix}.content",
                        generation.message.content,
                    )
                else:
                    span.set_attribute(
                        f"{prefix}.content",
                        json.dumps(generation.message.content, cls=CustomJsonEncode),
                    )
                if generation.generation_info.get("finish_reason"):
                    span.set_attribute(
                        f"{prefix}.finish_reason",
                        generation.generation_info.get("finish_reason"),
                    )

                if generation.message.additional_kwargs.get("function_call"):
                    span.set_attribute(
                        f"{prefix}.tool_calls.0.name",
                        generation.message.additional_kwargs.get("function_call").get(
                            "name"
                        ),
                    )
                    span.set_attribute(
                        f"{prefix}.tool_calls.0.arguments",
                        generation.message.additional_kwargs.get("function_call").get(
                            "arguments"
                        ),
                    )
            i += 1


class SyncSpanCallbackHandler(BaseCallbackHandler):
    def __init__(self, tracer: Tracer) -> None:
        super().__init__()
        self.tracer = tracer
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
        if "kwargs" in serialized and serialized["kwargs"].get("name"):
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
            context_api.attach(
                context_api.set_value("association_properties", metadata)
            )

        if parent_run_id is not None and parent_run_id in self.spans:
            span = self.tracer.start_span(
                span_name, context=self.spans[parent_run_id].context, kind=kind
            )
        else:
            span = self.tracer.start_span(span_name)

        span.set_attribute(SpanAttributes.TRACELOOP_WORKFLOW_NAME, workflow_name)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_PATH, entity_path)

        current_context = set_span_in_context(span)

        token = context_api.attach(
            context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True)
        )

        self.spans[run_id] = SpanHolder(span, token, current_context, [], workflow_name, entity_name, entity_path)

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
            metadata=metadata
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
            metadata
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
                    cls=CustomJsonEncode,
                ),
            )

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
        span = self._get_span(run_id)
        if should_send_prompts():
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                json.dumps(
                    {"outputs": outputs, "kwargs": kwargs}, cls=CustomJsonEncode
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
        name = self._get_name_from_callback(serialized, kwargs=kwargs)
        span = self._create_llm_span(
            run_id, parent_run_id, name, LLMRequestTypeValues.CHAT, metadata=metadata
        )
        _set_chat_request(span, serialized, messages, kwargs)

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
        name = self._get_name_from_callback(serialized, kwargs=kwargs)
        span = self._create_llm_span(
            run_id, parent_run_id, name, LLMRequestTypeValues.COMPLETION
        )
        _set_llm_request(span, serialized, prompts, kwargs)

    @dont_throw
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ):
        span = self._get_span(run_id)

        token_usage = (response.llm_output or {}).get("token_usage")
        if token_usage is not None:
            prompt_tokens = token_usage.get("prompt_tokens") or token_usage.get(
                "input_token_count"
            )
            completion_tokens = token_usage.get("completion_tokens") or token_usage.get(
                "generated_token_count"
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

        if response.llm_output is not None:
            model_name = response.llm_output.get(
                "model_name"
            ) or response.llm_output.get("model_id")
            if model_name is not None:
                span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, model_name)

        _set_chat_response(span, response)
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
        name = self._get_name_from_callback(serialized, kwargs=kwargs)
        workflow_name = self.get_workflow_name(parent_run_id)
        entity_path = self.get_entity_path(parent_run_id)

        span = self._create_task_span(
            run_id, parent_run_id, name, TraceloopSpanKindValues.TOOL, workflow_name, name, entity_path
        )
        if should_send_prompts():
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
        span = self._get_span(run_id)
        if should_send_prompts():
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                json.dumps({"output": output, "kwargs": kwargs}, cls=CustomJsonEncode),
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
        elif parent_span.entity_path == "" and parent_span.entity_name == parent_span.workflow_name:
            return ""
        elif parent_span.entity_path == "":
            return f"{parent_span.entity_name}"
        else:
            return f"{parent_span.entity_path}.{parent_span.entity_name}"
