import json

from opentelemetry.instrumentation.groq.utils import (
    dont_throw,
    model_as_dict,
    set_span_attribute,
    should_send_prompts,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_RESPONSE_ID,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import (
    SpanAttributes,
)

_GROQ_PROVIDER = GenAIAttributes.GenAiProviderNameValues.GROQ.value
_CHAT_OPERATION = GenAIAttributes.GenAiOperationNameValues.CHAT.value

# Groq API returns finish_reason as an OpenAI-compatible string.
# Map to OTel standard values; unknown / None → "".
# Note: Groq "tool_calls" (plural, OpenAI-compatible) → OTel "tool_call" (singular).
_FINISH_REASON_MAP = {
    "stop": "stop",
    "length": "length",
    "content_filter": "content_filter",
    "tool_calls": "tool_call",
}


def _map_groq_finish_reason(finish_reason):
    if not finish_reason:
        return ""
    return _FINISH_REASON_MAP.get(str(finish_reason), str(finish_reason))


def _collect_finish_reasons_from_response(response):
    if response is None:
        return []
    choices = getattr(response, "choices", None) or []
    return [_map_groq_finish_reason(getattr(c, "finish_reason", None)) for c in choices]


def _content_to_parts(content):
    """Convert OpenAI-compatible message content to OTel parts array."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "content": content}] if content else []
    # List of content blocks (multimodal)
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            parts.append({"type": "text", "content": block.get("text", "")})
        elif block_type == "image_url":
            url = (block.get("image_url") or {}).get("url", "")
            if url.startswith("data:"):
                try:
                    header, data = url.split(",", 1)
                    mime_type = header.split(":")[1].split(";")[0]
                    parts.append({"type": "blob", "modality": "image", "mime_type": mime_type, "content": data})
                except Exception:
                    parts.append({"type": "uri", "modality": "image", "uri": url})
            else:
                parts.append({"type": "uri", "modality": "image", "uri": url})
        else:
            parts.append({"type": block_type or "unknown", **{k: v for k, v in block.items() if k != "type"}})
    return parts


def _tool_calls_to_parts(tool_calls):
    """Convert OpenAI tool_calls list to OTel tool_call parts.

    Handles both dict representations (user kwargs) and object representations
    (e.g. Pydantic models returned by the Groq SDK).
    """
    parts = []
    for tc in tool_calls or []:
        if isinstance(tc, dict):
            tc_id = tc.get("id") or ""
            fn = tc.get("function") or {}
            fn_name = fn.get("name") or ""
            args_raw = fn.get("arguments")
        elif hasattr(tc, "function"):
            tc_id = getattr(tc, "id", "") or ""
            fn = tc.function
            fn_name = getattr(fn, "name", "") or ""
            args_raw = getattr(fn, "arguments", None)
        else:
            continue
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except (json.JSONDecodeError, TypeError):
                args = args_raw
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            args = None
        part = {"type": "tool_call", "name": fn_name}
        if tc_id:
            part["id"] = tc_id
        if args is not None:
            part["arguments"] = args
        parts.append(part)
    return parts


@dont_throw
def set_input_attributes(span, kwargs):
    if not span.is_recording() or not should_send_prompts():
        return

    messages = []
    for msg in kwargs.get("messages") or []:
        role = msg.get("role", "user")

        if role == "tool":
            parts = [
                {
                    "type": "tool_call_response",
                    "id": msg.get("tool_call_id") or "",
                    "response": msg.get("content") or "",
                }
            ]
        else:
            parts = _content_to_parts(msg.get("content"))
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                parts.extend(_tool_calls_to_parts(tool_calls))

        messages.append({"role": role, "parts": parts})

    if messages:
        set_span_attribute(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES, json.dumps(messages))


@dont_throw
def set_model_input_attributes(span, kwargs):
    if not span.is_recording():
        return

    set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MODEL, kwargs.get("model"))
    set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS, kwargs.get("max_tokens"))
    set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, kwargs.get("temperature"))
    set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_P, kwargs.get("top_p"))
    set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY,
        kwargs.get("frequency_penalty"),
    )
    set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY,
        kwargs.get("presence_penalty"),
    )
    set_span_attribute(span, SpanAttributes.GEN_AI_IS_STREAMING, kwargs.get("stream") or False)

    if should_send_prompts():
        tools = kwargs.get("tools")
        if tools:
            try:
                set_span_attribute(span, GenAIAttributes.GEN_AI_TOOL_DEFINITIONS, json.dumps(tools))
            except Exception:
                pass


def set_streaming_response_attributes(span, accumulated_content, finish_reason=None, tool_calls=None):
    """Set gen_ai.output.messages span attribute for accumulated streaming response."""
    if not span.is_recording() or not should_send_prompts():
        return

    mapped_reason = _map_groq_finish_reason(finish_reason)
    parts = [{"type": "text", "content": accumulated_content}] if accumulated_content else []
    if tool_calls:
        parts.extend(_tool_calls_to_parts(tool_calls))
    message = {"role": "assistant", "parts": parts, "finish_reason": mapped_reason}
    set_span_attribute(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, json.dumps([message]))


def set_model_streaming_response_attributes(span, usage, finish_reasons=None):
    if not span.is_recording():
        return

    if usage:
        set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, usage.completion_tokens)
        set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, usage.prompt_tokens)
        set_span_attribute(span, SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS, usage.total_tokens)

    if finish_reasons:
        mapped = [_map_groq_finish_reason(fr) for fr in finish_reasons]
        mapped = [m for m in mapped if m]
        if mapped:
            set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, mapped)


@dont_throw
def set_model_response_attributes(span, response, token_histogram):
    if not span.is_recording():
        return

    reasons = [r for r in _collect_finish_reasons_from_response(response) if r]
    if reasons:
        set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, reasons)

    response = model_as_dict(response)
    set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, response.get("model"))
    set_span_attribute(span, GEN_AI_RESPONSE_ID, response.get("id"))

    usage = response.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if usage:
        set_span_attribute(span, SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS, usage.get("total_tokens"))
        set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens)
        set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)

    if isinstance(prompt_tokens, int) and prompt_tokens >= 0 and token_histogram is not None:
        token_histogram.record(
            prompt_tokens,
            attributes={
                GenAIAttributes.GEN_AI_PROVIDER_NAME: _GROQ_PROVIDER,
                GenAIAttributes.GEN_AI_OPERATION_NAME: _CHAT_OPERATION,
                GenAIAttributes.GEN_AI_REQUEST_MODEL: response.get("model"),
                GenAIAttributes.GEN_AI_TOKEN_TYPE: "input",
                GenAIAttributes.GEN_AI_RESPONSE_MODEL: response.get("model"),
            },
        )

    if isinstance(completion_tokens, int) and completion_tokens >= 0 and token_histogram is not None:
        token_histogram.record(
            completion_tokens,
            attributes={
                GenAIAttributes.GEN_AI_PROVIDER_NAME: _GROQ_PROVIDER,
                GenAIAttributes.GEN_AI_OPERATION_NAME: _CHAT_OPERATION,
                GenAIAttributes.GEN_AI_REQUEST_MODEL: response.get("model"),
                GenAIAttributes.GEN_AI_TOKEN_TYPE: "output",
                GenAIAttributes.GEN_AI_RESPONSE_MODEL: response.get("model"),
            },
        )


def set_response_attributes(span, response):
    if not span.is_recording() or not should_send_prompts():
        return

    choices = model_as_dict(response).get("choices") or []
    messages = []

    for choice in choices:
        message = choice.get("message") or {}
        finish_reason = _map_groq_finish_reason(choice.get("finish_reason"))
        role = message.get("role") or "assistant"

        parts = _content_to_parts(message.get("content"))

        # tool_calls (modern OpenAI format)
        tool_calls = message.get("tool_calls")
        if tool_calls:
            parts.extend(_tool_calls_to_parts(tool_calls))

        # function_call (legacy OpenAI format)
        function_call = message.get("function_call")
        if function_call:
            args_raw = function_call.get("arguments")
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except (json.JSONDecodeError, TypeError):
                    args = args_raw
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                args = None
            part = {"type": "tool_call", "name": function_call.get("name") or ""}
            if args is not None:
                part["arguments"] = args
            parts.append(part)

        messages.append({"role": role, "parts": parts, "finish_reason": finish_reason})

    if messages:
        set_span_attribute(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, json.dumps(messages))
