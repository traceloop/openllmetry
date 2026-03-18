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
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.metrics import Histogram
from opentelemetry.semconv_ai import (
    SpanAttributes,
)
from opentelemetry.trace.span import Span


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


def _set_span_attribute(span: Span, key: str, value: Any) -> None:
    if value is not None:
        if value != "":
            span.set_attribute(key, value)
        else:
            span.set_attribute(key, "")


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

    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MODEL, model)
    # response is not available for LLM requests (as opposed to chat)
    _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, model)

    if "invocation_params" in kwargs:
        params = (
            kwargs["invocation_params"].get("params") or kwargs["invocation_params"]
        )
    else:
        params = kwargs

    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS,
        params.get("max_tokens") or params.get("max_new_tokens"),
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, params.get("temperature")
    )
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_P, params.get("top_p"))

    tools = kwargs.get("invocation_params", {}).get("tools", [])
    if tools:
        tool_defs = []
        for tool in tools:
            tool_function = tool.get("function", tool)
            tool_def = {
                "name": tool_function.get("name"),
                "description": tool_function.get("description"),
            }
            params = tool_function.get("parameters") or tool.get("input_schema")
            if params is not None:
                tool_def["parameters"] = params
            tool_defs.append(tool_def)
        span.set_attribute(GenAIAttributes.GEN_AI_TOOL_DEFINITIONS, json.dumps(tool_defs))


def set_llm_request(
    span: Span,
    serialized: dict[str, Any],
    prompts: list[str],
    kwargs: Any,
    span_holder: SpanHolder,
) -> None:
    set_request_params(span, kwargs, span_holder)

    if should_send_prompts():
        messages = [{"role": "user", "content": msg} for msg in prompts]
        span.set_attribute(GenAIAttributes.GEN_AI_INPUT_MESSAGES, json.dumps(messages))


def set_chat_request(
    span: Span,
    serialized: dict[str, Any],
    messages: list[list[BaseMessage]],
    kwargs: Any,
    span_holder: SpanHolder,
) -> None:
    set_request_params(span, serialized.get("kwargs", {}), span_holder)

    functions = kwargs.get("invocation_params", {}).get("functions", [])
    if functions:
        tool_defs = [
            {
                "name": f.get("name"),
                "description": f.get("description"),
                "parameters": f.get("parameters"),
            }
            for f in functions
        ]
        span.set_attribute(GenAIAttributes.GEN_AI_TOOL_DEFINITIONS, json.dumps(tool_defs))

    if should_send_prompts():
        input_messages = []
        for message in messages:
            for msg in message:
                msg_obj = {"role": _message_type_to_role(msg.type)}

                tool_calls = (
                    msg.tool_calls
                    if hasattr(msg, "tool_calls")
                    else msg.additional_kwargs.get("tool_calls")
                )
                if tool_calls:
                    msg_obj["tool_calls"] = _build_tool_calls_list(tool_calls)

                content = (
                    msg.content
                    if isinstance(msg.content, str)
                    else json.dumps(msg.content, cls=CallbackFilteredJSONEncoder)
                )
                if content:
                    msg_obj["content"] = content

                if msg.type == "tool" and hasattr(msg, "tool_call_id"):
                    msg_obj["tool_call_id"] = msg.tool_call_id

                input_messages.append(msg_obj)

        if input_messages:
            span.set_attribute(GenAIAttributes.GEN_AI_INPUT_MESSAGES, json.dumps(input_messages))


def set_chat_response(span: Span, response: LLMResult) -> None:
    if not should_send_prompts():
        return

    output_messages = []
    for generations in response.generations:
        for generation in generations:
            if hasattr(generation, "message") and generation.message and hasattr(generation.message, "type"):
                role = _message_type_to_role(generation.message.type)
            else:
                role = "assistant"

            msg_obj = {"role": role}

            # Try to get content from various sources
            content = None
            if hasattr(generation, "text") and generation.text:
                content = generation.text
            elif hasattr(generation, "message") and generation.message and generation.message.content:
                if isinstance(generation.message.content, str):
                    content = generation.message.content
                else:
                    content = json.dumps(generation.message.content, cls=CallbackFilteredJSONEncoder)

            if content:
                msg_obj["content"] = content

            # Set finish reason if available
            if generation.generation_info and generation.generation_info.get("finish_reason"):
                msg_obj["finish_reason"] = generation.generation_info.get("finish_reason")

            # Handle tool calls and function calls
            if hasattr(generation, "message") and generation.message:
                # Handle legacy function_call format (single function call)
                if generation.message.additional_kwargs.get("function_call"):
                    fc = generation.message.additional_kwargs.get("function_call")
                    msg_obj["role"] = "assistant"
                    msg_obj["tool_calls"] = [{"name": fc.get("name"), "arguments": fc.get("arguments")}]

                # Handle new tool_calls format (multiple tool calls)
                tool_calls = (
                    generation.message.tool_calls
                    if hasattr(generation.message, "tool_calls")
                    else generation.message.additional_kwargs.get("tool_calls")
                )
                if tool_calls and isinstance(tool_calls, list):
                    msg_obj["role"] = "assistant"
                    msg_obj["tool_calls"] = _build_tool_calls_list(tool_calls)

            output_messages.append(msg_obj)

    if output_messages:
        span.set_attribute(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages))


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

    # Early return if no generations to avoid potential issues
    if not response.generations:
        return

    try:
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
    except Exception:
        # If there's any issue processing usage metadata, continue without it
        pass

    if (
        input_tokens > 0
        or output_tokens > 0
        or total_tokens > 0
        or cache_read_tokens > 0
    ):
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
            input_tokens,
        )
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
            output_tokens,
        )
        _set_span_attribute(
            span,
            SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS,
            total_tokens,
        )
        _set_span_attribute(
            span,
            SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
            cache_read_tokens,
        )
        if record_token_usage:
            vendor = span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM, "langchain")

            if input_tokens > 0:
                token_histogram.record(
                    input_tokens,
                    attributes={
                        GenAIAttributes.GEN_AI_SYSTEM: vendor,
                        GenAIAttributes.GEN_AI_TOKEN_TYPE: "input",
                        GenAIAttributes.GEN_AI_RESPONSE_MODEL: model_name,
                    },
                )

            if output_tokens > 0:
                token_histogram.record(
                    output_tokens,
                    attributes={
                        GenAIAttributes.GEN_AI_SYSTEM: vendor,
                        GenAIAttributes.GEN_AI_TOKEN_TYPE: "output",
                        GenAIAttributes.GEN_AI_RESPONSE_MODEL: model_name,
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


def _build_tool_calls_list(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for tool_call in tool_calls:
        tool_call_dict = dict(tool_call)
        tool_id = tool_call_dict.get("id")
        tool_name = tool_call_dict.get(
            "name", tool_call_dict.get("function", {}).get("name")
        )
        tool_args = tool_call_dict.get(
            "args", tool_call_dict.get("function", {}).get("arguments")
        )

        call_obj = {}
        if tool_id:
            call_obj["id"] = tool_id
        if tool_name:
            call_obj["name"] = tool_name
        if tool_args is not None:
            call_obj["arguments"] = json.dumps(tool_args, cls=CallbackFilteredJSONEncoder)
        result.append(call_obj)
    return result
