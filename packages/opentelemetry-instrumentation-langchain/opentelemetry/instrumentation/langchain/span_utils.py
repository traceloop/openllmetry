import json
import logging
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

logger = logging.getLogger(__name__)


_FINISH_REASON_MAP = {
    # OpenAI / LangChain-native
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_call",
    "function_call": "tool_call",
    "content_filter": "content_filter",
    # Anthropic (surfaced through langchain-anthropic)
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_call",
    "max_tokens": "length",
}


def _map_finish_reason(reason):
    if not reason:
        return reason
    return _FINISH_REASON_MAP.get(reason, reason)


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
    elif message_type == "function":
        return "tool"
    else:
        return "unknown"


def _set_span_attribute(span: Span, key: str, value: Any) -> None:
    if value is not None:
        if value != "":
            span.set_attribute(key, value)
        else:
            span.set_attribute(key, "")


def _content_to_parts(content) -> list[dict]:
    """Convert LangChain message content (str or list-of-blocks) into OTel parts."""
    if isinstance(content, str):
        return [{"type": "text", "content": content}] if content else []
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                if block:
                    parts.append({"type": "text", "content": block})
            elif isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "text":
                    text = block.get("text", "")
                    if text:
                        parts.append({"type": "text", "content": text})
                elif block_type == "image_url":
                    url = (block.get("image_url") or {}).get("url", "")
                    if url:
                        parts.append({"type": "uri", "modality": "image", "uri": url})
                elif block_type == "image":
                    # base64 image block
                    parts.append({
                        "type": "blob",
                        "modality": "image",
                        "mime_type": block.get("media_type", block.get("mime_type", "")),
                        "content": block.get("data", ""),
                    })
                else:
                    # Unknown block type — preserve as text with JSON
                    parts.append({"type": "text", "content": json.dumps(block, cls=CallbackFilteredJSONEncoder)})
            else:
                parts.append({"type": "text", "content": str(block)})
        return parts
    # Fallback: single non-string value
    return [{"type": "text", "content": str(content)}] if content else []


def _tool_calls_to_parts(tool_calls) -> list[dict]:
    parts = []
    if not tool_calls:
        return parts
    for tc in tool_calls:
        tc_dict = dict(tc)
        tool_id = tc_dict.get("id", "")
        tool_name = tc_dict.get(
            "name", tc_dict.get("function", {}).get("name", "")
        )
        tool_args = tc_dict.get(
            "args", tc_dict.get("function", {}).get("arguments")
        )
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except (json.JSONDecodeError, TypeError):
                pass
        part = {
            "type": "tool_call",
            "id": tool_id,
            "name": tool_name,
        }
        if tool_args is not None:
            part["arguments"] = tool_args
        parts.append(part)
    return parts


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
            tool_def = {"name": tool_function.get("name")}
            if tool_function.get("description"):
                tool_def["description"] = tool_function.get("description")
            params_val = tool_function.get("parameters", tool.get("input_schema"))
            if params_val:
                tool_def["parameters"] = params_val
            tool_defs.append(tool_def)
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_TOOL_DEFINITIONS,
            json.dumps(tool_defs, cls=CallbackFilteredJSONEncoder),
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
        input_messages = []
        for msg in prompts:
            input_messages.append({
                "role": "user",
                "parts": [{"type": "text", "content": msg}],
            })
        if input_messages:
            _set_span_attribute(
                span,
                GenAIAttributes.GEN_AI_INPUT_MESSAGES,
                json.dumps(input_messages, cls=CallbackFilteredJSONEncoder),
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
        # Tool definitions from functions
        functions = kwargs.get("invocation_params", {}).get("functions", [])
        if functions:
            tool_defs = []
            for function in functions:
                tool_def = {"name": function.get("name")}
                if function.get("description"):
                    tool_def["description"] = function.get("description")
                if function.get("parameters"):
                    tool_def["parameters"] = function.get("parameters")
                tool_defs.append(tool_def)
            _set_span_attribute(
                span,
                GenAIAttributes.GEN_AI_TOOL_DEFINITIONS,
                json.dumps(tool_defs, cls=CallbackFilteredJSONEncoder),
            )

        input_messages = []
        system_instructions = []

        for message in messages:
            for msg in message:
                role = _message_type_to_role(msg.type)

                if role == "system":
                    system_instructions.extend(_content_to_parts(msg.content))
                    continue

                # Tool response (for tool messages)
                if role == "tool" and hasattr(msg, "tool_call_id"):
                    content_str = (
                        msg.content
                        if isinstance(msg.content, str)
                        else json.dumps(msg.content, cls=CallbackFilteredJSONEncoder)
                    )
                    parts = [{
                        "type": "tool_call_response",
                        "id": msg.tool_call_id,
                        "response": content_str,
                    }]
                else:
                    parts = _content_to_parts(msg.content)

                    # Tool calls (for assistant messages)
                    tool_calls = (
                        msg.tool_calls
                        if hasattr(msg, "tool_calls")
                        else msg.additional_kwargs.get("tool_calls")
                    )
                    if tool_calls:
                        parts.extend(_tool_calls_to_parts(tool_calls))

                msg_obj = {"role": role, "parts": parts}
                input_messages.append(msg_obj)

        if system_instructions:
            _set_span_attribute(
                span,
                GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS,
                json.dumps(system_instructions, cls=CallbackFilteredJSONEncoder),
            )
        if input_messages:
            _set_span_attribute(
                span,
                GenAIAttributes.GEN_AI_INPUT_MESSAGES,
                json.dumps(input_messages, cls=CallbackFilteredJSONEncoder),
            )


def set_chat_response(span: Span, response: LLMResult) -> None:
    send_prompts = should_send_prompts()

    output_messages = []
    finish_reasons = []

    for generations in response.generations:
        for generation in generations:
            # Finish reason — always collect (metadata, not content)
            fr = None
            if generation.generation_info and generation.generation_info.get("finish_reason"):
                fr = _map_finish_reason(generation.generation_info["finish_reason"])
                if fr not in finish_reasons:
                    finish_reasons.append(fr)

            if not send_prompts:
                continue

            if hasattr(generation, "message") and generation.message and hasattr(generation.message, "type"):
                role = _message_type_to_role(generation.message.type)
            else:
                role = "assistant"

            parts = []

            # Try to get content from various sources
            if hasattr(generation, "text") and generation.text:
                parts.append({"type": "text", "content": generation.text})
            elif hasattr(generation, "message") and generation.message and generation.message.content:
                parts.extend(_content_to_parts(generation.message.content))

            # Handle tool calls and function calls
            if hasattr(generation, "message") and generation.message:
                # Handle legacy function_call format (single function call)
                fc = generation.message.additional_kwargs.get("function_call")
                if fc:
                    fc_args = fc.get("arguments")
                    if isinstance(fc_args, str):
                        try:
                            fc_args = json.loads(fc_args)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    parts.append({
                        "type": "tool_call",
                        "id": "",
                        "name": fc.get("name"),
                        "arguments": fc_args,
                    })

                # Handle new tool_calls format (multiple tool calls)
                tool_calls = (
                    generation.message.tool_calls
                    if hasattr(generation.message, "tool_calls")
                    else generation.message.additional_kwargs.get("tool_calls")
                )
                if tool_calls and isinstance(tool_calls, list):
                    parts.extend(_tool_calls_to_parts(tool_calls))

            msg_obj = {"role": role, "parts": parts, "finish_reason": fr if fr else ""}
            output_messages.append(msg_obj)

    if output_messages:
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps(output_messages, cls=CallbackFilteredJSONEncoder),
        )
    if finish_reasons:
        span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)


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
    except Exception as e:
        # If there's any issue processing usage metadata, continue without it
        logger.warning("Error processing usage metadata: %s", e)
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
            vendor = span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME, "langchain")

            if input_tokens > 0:
                token_histogram.record(
                    input_tokens,
                    attributes={
                        GenAIAttributes.GEN_AI_PROVIDER_NAME: vendor,
                        GenAIAttributes.GEN_AI_TOKEN_TYPE: "input",
                        GenAIAttributes.GEN_AI_RESPONSE_MODEL: model_name,
                    },
                )

            if output_tokens > 0:
                token_histogram.record(
                    output_tokens,
                    attributes={
                        GenAIAttributes.GEN_AI_PROVIDER_NAME: vendor,
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
