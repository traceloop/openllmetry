import json
import logging

from opentelemetry.instrumentation.anthropic.config import Config
from opentelemetry.instrumentation.anthropic.utils import (
    JSONEncoder,
    dont_throw,
    model_as_dict,
    should_send_prompts,
    _extract_response_data,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

logger = logging.getLogger(__name__)

# Anthropic stop_reason -> OTel GenAI FinishReason enum
_FINISH_REASON_MAP = {
    "end_turn": "stop",
    "tool_use": "tool_call",
    "max_tokens": "length",
    "stop_sequence": "stop",
}


def _map_finish_reason(anthropic_reason):
    """Map an Anthropic stop_reason to the OTel GenAI FinishReason enum value."""
    if not anthropic_reason:
        return ""
    return _FINISH_REASON_MAP.get(anthropic_reason, anthropic_reason)


async def _process_image_item(item, trace_id, span_id, message_index, content_index):
    source = item.get("source", {})
    media_type = source.get("media_type", "image/unknown")

    if not Config.upload_base64_image:
        return {
            "type": "blob",
            "modality": "image",
            "mime_type": media_type,
            "content": source.get("data", ""),
        }

    image_format = media_type.split("/")[1]
    image_name = f"message_{message_index}_content_{content_index}.{image_format}"
    base64_string = source.get("data")
    url = await Config.upload_base64_image(trace_id, span_id, image_name, base64_string)

    return {"type": "uri", "modality": "image", "uri": url}


async def _content_to_parts(message_index, content, span):
    """Convert Anthropic message content into OTel spec parts array.

    Returns a list of parts following the gen-ai-input-messages.json schema:
      - Text:      {"type": "text", "content": "..."}
      - Reasoning: {"type": "reasoning", "content": "..."}
      - ToolCall:  {"type": "tool_call", "id": "...", "name": "...", "arguments": {...}}
      - Image:     {"type": "blob", ...} or {"type": "uri", ...}
    """
    if isinstance(content, str):
        return [{"type": "text", "content": content}]
    elif isinstance(content, list):
        parts = []
        for j, item in enumerate(content):
            item_dict = model_as_dict(item) if not isinstance(item, dict) else dict(item)
            block_type = item_dict.get("type")

            if block_type == "tool_use":
                tool_input = item_dict.get("input")
                if isinstance(tool_input, str):
                    try:
                        tool_input = json.loads(tool_input)
                    except (json.JSONDecodeError, TypeError):
                        pass
                parts.append({
                    "type": "tool_call",
                    "id": item_dict.get("id"),
                    "name": item_dict.get("name"),
                    "arguments": tool_input,
                })
            elif block_type == "thinking":
                parts.append({
                    "type": "reasoning",
                    "content": item_dict.get("thinking", ""),
                })
            elif block_type == "redacted_thinking":
                continue
            elif block_type == "text":
                text_value = item_dict.get("text", "")
                parts.append({"type": "text", "content": text_value})
            elif block_type == "image":
                source = item_dict.get("source", {})
                src_type = source.get("type")
                if src_type == "base64":
                    processed = await _process_image_item(
                        item_dict,
                        span.context.trace_id,
                        span.context.span_id,
                        message_index,
                        j,
                    )
                    parts.append(processed)
                elif src_type == "url":
                    parts.append({"type": "uri", "modality": "image", "uri": source.get("url", "")})
                else:
                    parts.append(item_dict)
            elif block_type == "tool_result":
                parts.append({
                    "type": "tool_call_response",
                    "id": item_dict.get("tool_use_id"),
                    "response": item_dict.get("content"),
                })
            else:
                parts.append(item_dict)
        return parts
    return []


@dont_throw
async def aset_input_attributes(span, kwargs):
    from opentelemetry.instrumentation.anthropic import set_span_attribute

    set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MODEL, kwargs.get("model"))
    set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS, kwargs.get("max_tokens_to_sample") or kwargs.get("max_tokens")
    )
    set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, kwargs.get("temperature")
    )
    set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_P, kwargs.get("top_p"))
    set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY, kwargs.get("frequency_penalty")
    )
    set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY, kwargs.get("presence_penalty")
    )
    set_span_attribute(span, SpanAttributes.GEN_AI_IS_STREAMING, kwargs.get("stream"))

    if should_send_prompts():
        if kwargs.get("prompt") is not None:
            set_span_attribute(
                span,
                GenAIAttributes.GEN_AI_INPUT_MESSAGES,
                json.dumps([{
                    "role": "user",
                    "parts": [{"type": "text", "content": kwargs.get("prompt")}],
                }]),
            )

        elif kwargs.get("messages") is not None:
            if kwargs.get("system"):
                set_span_attribute(
                    span,
                    GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS,
                    await _dump_system_content(
                        message_index=0, span=span, content=kwargs.get("system")
                    ),
                )

            input_messages = []
            for i, message in enumerate(kwargs.get("messages")):
                content = message.get("content")
                parts = await _content_to_parts(
                    message_index=i, content=content, span=span
                )
                msg_obj = {
                    "role": message.get("role"),
                    "parts": parts,
                }
                input_messages.append(msg_obj)

            set_span_attribute(
                span,
                GenAIAttributes.GEN_AI_INPUT_MESSAGES,
                json.dumps(input_messages, cls=JSONEncoder),
            )

        if kwargs.get("tools") is not None:
            tool_defs = []
            for tool in kwargs.get("tools"):
                tool_def = {"name": tool.get("name")}
                if tool.get("description"):
                    tool_def["description"] = tool.get("description")
                if tool.get("input_schema") is not None:
                    tool_def["input_schema"] = tool.get("input_schema")
                tool_defs.append(tool_def)
            set_span_attribute(
                span,
                GenAIAttributes.GEN_AI_TOOL_DEFINITIONS,
                json.dumps(tool_defs, cls=JSONEncoder),
            )

        output_format = kwargs.get("output_format")
        if output_format and isinstance(output_format, dict):
            if output_format.get("type") == "json_schema":
                schema = output_format.get("schema")
                if schema:
                    set_span_attribute(
                        span,
                        SpanAttributes.GEN_AI_REQUEST_STRUCTURED_OUTPUT_SCHEMA,
                        json.dumps(schema),
                    )


async def _dump_system_content(message_index, content, span):
    """Convert system content to a JSON array of OTel spec parts for gen_ai.system_instructions.

    Returns a JSON string containing a flat array of typed parts:
      - Text:  {"type": "text", "content": "..."}
      - Image: {"type": "uri", ...} or {"type": "blob", ...}
    """
    if isinstance(content, str):
        return json.dumps([{"type": "text", "content": content}])
    elif isinstance(content, list):
        parts = []
        for j, item in enumerate(content):
            item_dict = model_as_dict(item) if not isinstance(item, dict) else dict(item)
            block_type = item_dict.get("type")
            if block_type == "text":
                parts.append({"type": "text", "content": item_dict.get("text", "")})
            elif block_type == "image":
                source = item_dict.get("source", {})
                src_type = source.get("type")
                if src_type == "base64":
                    processed = await _process_image_item(
                        item_dict,
                        span.context.trace_id,
                        span.context.span_id,
                        message_index,
                        j,
                    )
                    parts.append(processed)
                elif src_type == "url":
                    parts.append({"type": "uri", "modality": "image", "uri": source.get("url", "")})
                else:
                    parts.append(item_dict)
            else:
                parts.append(item_dict)
        return json.dumps(parts, cls=JSONEncoder)


def _build_output_messages_from_content(response):
    """Build OTel spec-compliant output messages from an Anthropic response.

    Returns a list with a single assistant message using parts structure:
      - Text:      {"type": "text", "content": "..."}
      - ToolCall:  {"type": "tool_call", "id": "...", "name": "...", "arguments": {...}}
      - Reasoning: {"type": "reasoning", "content": "..."}
    """
    if response.get("completion"):
        msg = {
            "role": response.get("role", "assistant"),
            "parts": [{"type": "text", "content": response.get("completion")}],
        }
        msg["finish_reason"] = _map_finish_reason(response.get("stop_reason"))
        return [msg]

    if not response.get("content"):
        return []

    parts = []
    for content_block in response.get("content"):
        content_block_type = content_block.type
        if content_block_type == "text" and hasattr(content_block, "text"):
            parts.append({"type": "text", "content": content_block.text})
        elif content_block_type == "thinking":
            parts.append({
                "type": "reasoning",
                "content": getattr(content_block, "thinking", None),
            })
        elif content_block_type == "tool_use":
            tool_arguments = getattr(content_block, "input", None)
            if isinstance(tool_arguments, str):
                try:
                    tool_arguments = json.loads(tool_arguments)
                except (json.JSONDecodeError, TypeError):
                    pass
            parts.append({
                "type": "tool_call",
                "id": getattr(content_block, "id", None),
                "name": getattr(content_block, "name", None),
                "arguments": tool_arguments,
            })

    if not parts:
        return []

    msg = {
        "role": response.get("role", "assistant"),
        "parts": parts,
    }
    msg["finish_reason"] = _map_finish_reason(response.get("stop_reason"))
    return [msg]


async def _aset_span_completions(span, response):
    from opentelemetry.instrumentation.anthropic import set_span_attribute
    from opentelemetry.instrumentation.anthropic.utils import _aextract_response_data

    response = await _aextract_response_data(response)
    stop_reason = response.get("stop_reason")

    if stop_reason:
        span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, [_map_finish_reason(stop_reason)])

    if not should_send_prompts():
        return

    output_messages = _build_output_messages_from_content(response)

    if output_messages:
        set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps(output_messages, cls=JSONEncoder),
        )


def _set_span_completions(span, response):
    from opentelemetry.instrumentation.anthropic import set_span_attribute

    response = _extract_response_data(response)
    stop_reason = response.get("stop_reason")

    if stop_reason:
        span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, [_map_finish_reason(stop_reason)])

    if not should_send_prompts():
        return

    output_messages = _build_output_messages_from_content(response)

    if output_messages:
        set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps(output_messages, cls=JSONEncoder),
        )


@dont_throw
async def aset_response_attributes(span, response):
    from opentelemetry.instrumentation.anthropic import set_span_attribute
    from opentelemetry.instrumentation.anthropic.utils import _aextract_response_data

    response = await _aextract_response_data(response)
    set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, response.get("model"))
    set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, response.get("id"))

    if response.get("usage"):
        prompt_tokens = response.get("usage").input_tokens
        completion_tokens = response.get("usage").output_tokens
        set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)
        set_span_attribute(
            span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens
        )
        set_span_attribute(
            span,
            SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS,
            prompt_tokens + completion_tokens,
        )

    await _aset_span_completions(span, response)


@dont_throw
def set_response_attributes(span, response):
    from opentelemetry.instrumentation.anthropic import set_span_attribute

    response = _extract_response_data(response)
    set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, response.get("model"))
    set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, response.get("id"))

    if response.get("usage"):
        prompt_tokens = response.get("usage").input_tokens
        completion_tokens = response.get("usage").output_tokens
        set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)
        set_span_attribute(
            span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens
        )
        set_span_attribute(
            span,
            SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS,
            prompt_tokens + completion_tokens,
        )

    _set_span_completions(span, response)


@dont_throw
def set_streaming_response_attributes(span, complete_response_events):
    from opentelemetry.instrumentation.anthropic import set_span_attribute

    if not span.is_recording() or not complete_response_events:
        return

    # Collect all parts and determine finish_reason
    parts = []
    finish_reasons = []

    for event in complete_response_events:
        finish_reason = event.get("finish_reason")
        if finish_reason:
            mapped = _map_finish_reason(finish_reason)
            if mapped not in finish_reasons:
                finish_reasons.append(mapped)

        if should_send_prompts():
            if event.get("type") == "thinking":
                parts.append({
                    "type": "reasoning",
                    "content": event.get("text"),
                })
            elif event.get("type") == "tool_use":
                tool_arguments = event.get("input")
                if isinstance(tool_arguments, str):
                    try:
                        tool_arguments = json.loads(tool_arguments)
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif tool_arguments is None:
                    tool_arguments = None
                parts.append({
                    "type": "tool_call",
                    "id": event.get("id"),
                    "name": event.get("name"),
                    "arguments": tool_arguments,
                })
            elif event.get("type") == "text":
                parts.append({
                    "type": "text",
                    "content": event.get("text"),
                })

    if finish_reasons:
        span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)

    if parts:
        # Consolidate all parts into a single assistant message
        msg = {
            "role": "assistant",
            "parts": parts,
        }
        msg["finish_reason"] = finish_reasons[-1] if finish_reasons else ""
        output_messages = [msg]
        set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps(output_messages, cls=JSONEncoder),
        )
