import json

from opentelemetry.instrumentation.together.utils import (
    dont_throw,
    enum_value,
    model_as_dict,
    set_span_attribute,
    should_send_prompts,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes

# Together returns OpenAI-compatible finish reasons plus "eos". Map them to the
# OTel GenAI values; unknown values pass through unchanged, None becomes "".
_FINISH_REASON_MAP = {
    "stop": "stop",
    "eos": "stop",
    "length": "length",
    "content_filter": "content_filter",
    "tool_calls": "tool_call",
    "function_call": "tool_call",
    "error": "error",
}


def map_finish_reason(finish_reason):
    finish_reason = enum_value(finish_reason)
    if not finish_reason:
        return ""
    return _FINISH_REASON_MAP.get(str(finish_reason), str(finish_reason))


def _media_url_part(modality, url):
    if url.startswith("data:"):
        try:
            header, data = url.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
            return {"type": "blob", "modality": modality, "mime_type": mime_type, "content": data}
        except Exception:
            pass
    return {"type": "uri", "modality": modality, "uri": url}


_MEDIA_BLOCKS = {"image_url": "image", "video_url": "video", "audio_url": "audio"}


def _content_to_parts(content):
    """Convert OpenAI-compatible message content to OTel message parts."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "content": content}] if content else []

    parts = []
    for block in content:
        block = model_as_dict(block)
        if not isinstance(block, dict):
            continue
        block_type = enum_value(block.get("type"))
        if block_type == "text":
            parts.append({"type": "text", "content": block.get("text") or ""})
        elif block_type in _MEDIA_BLOCKS:
            url = (block.get(block_type) or {}).get("url") or ""
            parts.append(_media_url_part(_MEDIA_BLOCKS[block_type], url))
        else:
            parts.append({"type": block_type or "unknown", **{k: v for k, v in block.items() if k != "type"}})
    return parts


def _parse_arguments(arguments):
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return arguments
    return arguments


def _tool_calls_to_parts(tool_calls):
    """Convert OpenAI-compatible tool calls (dicts or SDK models) to OTel tool_call parts."""
    parts = []
    for tool_call in tool_calls or []:
        tool_call = model_as_dict(tool_call)
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        part = {"type": "tool_call", "name": function.get("name") or ""}
        if tool_call.get("id"):
            part["id"] = tool_call["id"]
        arguments = _parse_arguments(function.get("arguments"))
        if arguments is not None:
            part["arguments"] = arguments
        parts.append(part)
    return parts


def _input_message(message):
    message = model_as_dict(message)
    role = enum_value(message.get("role")) or "user"
    if role == "tool":
        parts = [
            {
                "type": "tool_call_response",
                "id": message.get("tool_call_id") or "",
                "response": message.get("content") or "",
            }
        ]
    else:
        parts = _content_to_parts(message.get("content"))
        parts.extend(_tool_calls_to_parts(message.get("tool_calls")))
    return {"role": role, "parts": parts}


def _output_message(choice, llm_request_type):
    finish_reason = map_finish_reason(choice.get("finish_reason"))
    if llm_request_type == LLMRequestTypeValues.COMPLETION:
        return {
            "role": "assistant",
            "parts": _content_to_parts(choice.get("text")),
            "finish_reason": finish_reason,
        }

    message = choice.get("message") or {}
    parts = []
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        parts.append({"type": "reasoning", "content": reasoning})
    parts.extend(_content_to_parts(message.get("content")))
    parts.extend(_tool_calls_to_parts(message.get("tool_calls")))
    return {
        "role": enum_value(message.get("role")) or "assistant",
        "parts": parts,
        "finish_reason": finish_reason,
    }


@dont_throw
def set_input_attributes(span, llm_request_type, kwargs):
    """Set gen_ai.input.messages and gen_ai.tool.definitions (content, so gated)."""
    if not span.is_recording() or not should_send_prompts():
        return

    if llm_request_type == LLMRequestTypeValues.CHAT:
        messages = [_input_message(message) for message in kwargs.get("messages") or []]
    elif llm_request_type == LLMRequestTypeValues.COMPLETION:
        prompt = kwargs.get("prompt")
        messages = [{"role": "user", "parts": _content_to_parts(prompt)}] if prompt else []
    else:
        messages = []

    if messages:
        set_span_attribute(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES, json.dumps(messages))

    tools = kwargs.get("tools")
    if tools:
        set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_TOOL_DEFINITIONS,
            json.dumps([model_as_dict(tool) for tool in tools]),
        )


@dont_throw
def set_model_input_attributes(span, kwargs):
    if not span.is_recording():
        return

    set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MODEL, kwargs.get("model"))
    set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS, kwargs.get("max_tokens"))
    set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, kwargs.get("temperature"))
    set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_P, kwargs.get("top_p"))
    set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_K, kwargs.get("top_k"))
    set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY, kwargs.get("frequency_penalty"))
    set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY, kwargs.get("presence_penalty"))
    set_span_attribute(span, SpanAttributes.GEN_AI_REQUEST_REPETITION_PENALTY, kwargs.get("repetition_penalty"))
    set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_SEED, kwargs.get("seed"))
    set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_CHOICE_COUNT, kwargs.get("n"))

    stop = kwargs.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    if stop:
        set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES, list(stop))

    set_span_attribute(span, SpanAttributes.GEN_AI_IS_STREAMING, bool(kwargs.get("stream")))


@dont_throw
def set_output_attributes(span, llm_request_type, response):
    """Set gen_ai.output.messages (content, so gated)."""
    if not span.is_recording() or not should_send_prompts():
        return

    choices = model_as_dict(response).get("choices") or []
    messages = [_output_message(model_as_dict(choice), llm_request_type) for choice in choices]
    if messages:
        set_span_attribute(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, json.dumps(messages))


@dont_throw
def set_model_response_attributes(span, response):
    if not span.is_recording():
        return

    response = model_as_dict(response)
    set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, response.get("model"))
    set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, response.get("id"))

    finish_reasons = [map_finish_reason(choice.get("finish_reason")) for choice in response.get("choices") or []]
    finish_reasons = [reason for reason in finish_reasons if reason]
    if finish_reasons:
        set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)

    usage = response.get("usage") or {}
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
    set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
    set_span_attribute(span, SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS, total_tokens)
    set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
    )
    set_span_attribute(
        span,
        SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS,
        (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
    )
