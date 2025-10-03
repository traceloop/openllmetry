import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from langchain_core.messages import (
    BaseMessage,
)
from langchain_core.outputs import (
    LLMResult,
)
from opentelemetry.context.context import Context
from opentelemetry.instrumentation.langchain.utils import (
    CallbackFilteredJSONEncoder,
    should_send_prompts,
)
from opentelemetry.metrics import Histogram
from opentelemetry.semconv_ai import (
    SpanAttributes,
)
from opentelemetry.trace.span import Span
from opentelemetry.util.types import AttributeValue


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


def _set_span_attribute(span: Span, name: str, value: AttributeValue):
    if value is not None and value != "":
        span.set_attribute(name, value)


def set_request_params(span, kwargs, span_holder: SpanHolder):
    if not span.is_recording():
        return

    for model_tag in ("model", "model_id", "model_name"):
        if (model := kwargs.get(model_tag)) is not None:
            span_holder.request_model = model
            break
        elif (
            model := (kwargs.get("invocation_params") or {}).get(model_tag)
        ) is not None:
            span_holder.request_model = model
            break
    else:
        model = "unknown"

    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, model)
    # response is not available for LLM requests (as opposed to chat)
    _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, model)

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

    tools = kwargs.get("invocation_params", {}).get("tools", [])
    for i, tool in enumerate(tools):
        tool_function = tool.get("function", tool)
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}.name",
            tool_function.get("name"),
        )
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}.description",
            tool_function.get("description"),
        )
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}.parameters",
            json.dumps(tool_function.get("parameters", tool.get("input_schema"))),
        )


def set_llm_request(
    span: Span,
    serialized: dict[str, Any],
    prompts: list[str],
    kwargs: Any,
    span_holder: SpanHolder,
) -> None:
    set_request_params(span, kwargs, span_holder)

    if should_send_prompts():
        for i, msg in enumerate(prompts):
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.{i}.role",
                "user",
            )
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                msg,
            )


def set_chat_request(
    span: Span,
    serialized: dict[str, Any],
    messages: list[list[BaseMessage]],
    kwargs: Any,
    span_holder: SpanHolder,
) -> None:
    set_request_params(span, serialized.get("kwargs", {}), span_holder)

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
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{i}.role",
                    _message_type_to_role(msg.type),
                )
                tool_calls = (
                    msg.tool_calls
                    if hasattr(msg, "tool_calls")
                    else msg.additional_kwargs.get("tool_calls")
                )

                if tool_calls:
                    _set_chat_tool_calls(
                        span, f"{SpanAttributes.LLM_PROMPTS}.{i}", tool_calls
                    )

                # Always set content if it exists, regardless of tool_calls presence
                content = (
                    msg.content
                    if isinstance(msg.content, str)
                    else json.dumps(msg.content, cls=CallbackFilteredJSONEncoder)
                )
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                    content,
                )

                if msg.type == "tool" and hasattr(msg, "tool_call_id"):
                    _set_span_attribute(
                        span,
                        f"{SpanAttributes.LLM_PROMPTS}.{i}.tool_call_id",
                        msg.tool_call_id,
                    )

                i += 1


def set_chat_response(span: Span, response: LLMResult) -> None:
    if not should_send_prompts():
        return

    i = 0
    for generations in response.generations:
        for generation in generations:
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{i}"
            if hasattr(generation, "text") and generation.text != "":
                _set_span_attribute(
                    span,
                    f"{prefix}.content",
                    generation.text,
                )
                _set_span_attribute(span, f"{prefix}.role", "assistant")
            else:
                _set_span_attribute(
                    span,
                    f"{prefix}.role",
                    _message_type_to_role(generation.type),
                )
                if generation.message.content is str:
                    _set_span_attribute(
                        span,
                        f"{prefix}.content",
                        generation.message.content,
                    )
                else:
                    _set_span_attribute(
                        span,
                        f"{prefix}.content",
                        json.dumps(
                            generation.message.content, cls=CallbackFilteredJSONEncoder
                        ),
                    )
                if generation.generation_info.get("finish_reason"):
                    _set_span_attribute(
                        span,
                        f"{prefix}.finish_reason",
                        generation.generation_info.get("finish_reason"),
                    )

                if generation.message.additional_kwargs.get("function_call"):
                    _set_span_attribute(
                        span,
                        f"{prefix}.tool_calls.0.name",
                        generation.message.additional_kwargs.get("function_call").get(
                            "name"
                        ),
                    )
                    _set_span_attribute(
                        span,
                        f"{prefix}.tool_calls.0.arguments",
                        generation.message.additional_kwargs.get("function_call").get(
                            "arguments"
                        ),
                    )

            if hasattr(generation, "message"):
                tool_calls = (
                    generation.message.tool_calls
                    if hasattr(generation.message, "tool_calls")
                    else generation.message.additional_kwargs.get("tool_calls")
                )
                if tool_calls and isinstance(tool_calls, list):
                    _set_span_attribute(
                        span,
                        f"{prefix}.role",
                        "assistant",
                    )
                    _set_chat_tool_calls(span, prefix, tool_calls)
            i += 1


def set_chat_response_usage(
    span: Span,
    response: LLMResult,
    token_histogram: Histogram,
    record_token_usage: bool,
    model_name: str
) -> None:
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    cache_read_tokens = 0

    for generations in response.generations:
        for generation in generations:
            if (
                hasattr(generation, "message")
                and hasattr(generation.message, "usage_metadata")
                and generation.message.usage_metadata is not None
            ):
                input_tokens += (
                    generation.message.usage_metadata.get("input_tokens")
                    or generation.message.usage_metadata.get("prompt_tokens")
                    or 0
                )
                output_tokens += (
                    generation.message.usage_metadata.get("output_tokens")
                    or generation.message.usage_metadata.get("completion_tokens")
                    or 0
                )
                total_tokens = input_tokens + output_tokens

                if generation.message.usage_metadata.get("input_token_details"):
                    input_token_details = generation.message.usage_metadata.get(
                        "input_token_details", {}
                    )
                    cache_read_tokens += input_token_details.get("cache_read", 0)

    if (
        input_tokens > 0
        or output_tokens > 0
        or total_tokens > 0
        or cache_read_tokens > 0
    ):
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
            input_tokens,
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            output_tokens,
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            total_tokens,
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
            cache_read_tokens,
        )
        if record_token_usage:
            vendor = span.attributes.get(SpanAttributes.LLM_SYSTEM, "Langchain")

            if input_tokens > 0:
                token_histogram.record(
                    input_tokens,
                    attributes={
                        SpanAttributes.LLM_SYSTEM: vendor,
                        SpanAttributes.LLM_TOKEN_TYPE: "input",
                        SpanAttributes.LLM_RESPONSE_MODEL: model_name,
                    },
                )

            if output_tokens > 0:
                token_histogram.record(
                    output_tokens,
                    attributes={
                        SpanAttributes.LLM_SYSTEM: vendor,
                        SpanAttributes.LLM_TOKEN_TYPE: "output",
                        SpanAttributes.LLM_RESPONSE_MODEL: model_name,
                    },
                )


def extract_model_name_from_response_metadata(response: LLMResult) -> str:
    for generations in response.generations:
        for generation in generations:
            if (
                getattr(generation, "message", None)
                and getattr(generation.message, "response_metadata", None)
                and (model_name := generation.message.response_metadata.get("model_name"))
            ):
                return model_name


def _extract_model_name_from_association_metadata(metadata: Optional[dict[str, Any]] = None) -> str:
    if metadata:
        return metadata.get("ls_model_name") or "unknown"
    return "unknown"


def _set_chat_tool_calls(
    span: Span, prefix: str, tool_calls: list[dict[str, Any]]
) -> None:
    for idx, tool_call in enumerate(tool_calls):
        tool_call_prefix = f"{prefix}.tool_calls.{idx}"
        tool_call_dict = dict(tool_call)
        tool_id = tool_call_dict.get("id")
        tool_name = tool_call_dict.get(
            "name", tool_call_dict.get("function", {}).get("name")
        )
        tool_args = tool_call_dict.get(
            "args", tool_call_dict.get("function", {}).get("arguments")
        )

        _set_span_attribute(span, f"{tool_call_prefix}.id", tool_id)
        _set_span_attribute(
            span,
            f"{tool_call_prefix}.name",
            tool_name,
        )
        _set_span_attribute(
            span,
            f"{tool_call_prefix}.arguments",
            json.dumps(tool_args, cls=CallbackFilteredJSONEncoder),
        )
