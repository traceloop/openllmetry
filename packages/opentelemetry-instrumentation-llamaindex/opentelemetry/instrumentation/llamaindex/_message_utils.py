"""Pure functions for building OTel GenAI semconv-compliant message JSON."""

import json
from typing import Any, Dict, List, Optional

# Finish reason mapping: covers OpenAI, Cohere, Anthropic, Google Gemini.
# OTel spec uses "tool_call" (singular) — OpenAI's "tool_calls" (plural) must be mapped.
_FINISH_REASON_MAP = {
    # OpenAI
    "tool_calls": "tool_call",
    "function_call": "tool_call",
    # Cohere
    "COMPLETE": "stop",
    "MAX_TOKENS": "length",
    "ERROR": "error",
    "ERROR_TOXIC": "content_filter",
    # Anthropic
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_call",
    "max_tokens": "length",
    # Google Gemini
    "STOP": "stop",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "FINISH_REASON_UNSPECIFIED": "error",
    "OTHER": "error",
}


def map_finish_reason(reason: Optional[str]) -> Optional[str]:
    """Map provider finish_reason to OTel enum value.

    Returns None if reason is None or empty (callers for top-level attr should omit).
    Returns mapped OTel value or pass-through for unmapped values.
    For per-message finish_reason, callers MUST apply fallback:
    ``map_finish_reason(r) or ""``.
    """
    if not reason:
        return None
    return _FINISH_REASON_MAP.get(reason, reason)


def _parse_arguments(arguments: Any) -> Any:
    """Parse tool call arguments to an object. Best-effort json.loads with fallback."""
    if arguments is None:
        return None
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return arguments
    return arguments


def _content_to_parts(content: Any) -> List[Dict]:
    """Convert LlamaIndex message content to OTel parts array.

    Handles: str/None → single TextPart or empty, list of content blocks → mapped by type.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "content": content}] if content else []
    if isinstance(content, list):
        return [_block_to_part(block) for block in content]
    return [{"type": "text", "content": str(content)}]


def _block_to_part(block: Any) -> Dict:
    """Convert a single content block to an OTel part dict."""
    if isinstance(block, str):
        return {"type": "text", "content": block}
    if not isinstance(block, dict):
        return {"type": "text", "content": str(block)}

    block_type = block.get("type", "")
    if block_type == "text":
        return {"type": "text", "content": block.get("content", block.get("text", ""))}
    if block_type in ("thinking", "reasoning"):
        return {"type": "reasoning", "content": block.get("thinking", block.get("content", block.get("text", "")))}
    if block_type == "image_url":
        url = block.get("image_url", {}).get("url", "")
        return {"type": "uri", "modality": "image", "uri": url}
    if block_type == "image":
        return _image_block_to_part(block)

    # Fallback: treat as text if it has recognizable content
    if "text" in block:
        return {"type": "text", "content": block["text"]}
    if "content" in block:
        return {"type": "text", "content": str(block["content"])}
    return {"type": "text", "content": str(block)}


def _image_block_to_part(block: Dict) -> Dict:
    """Convert an image content block to BlobPart or UriPart."""
    source = block.get("source", {})
    if source.get("type") == "base64":
        return {
            "type": "blob",
            "modality": "image",
            "mime_type": source.get("media_type", ""),
            "content": source.get("data", ""),
        }
    if source.get("type") == "url":
        return {"type": "uri", "modality": "image", "uri": source.get("url", "")}
    return {"type": "text", "content": str(block)}


def _extract_tool_calls(msg: Any) -> List[Dict]:
    """Extract tool_call parts from a LlamaIndex ChatMessage's additional_kwargs."""
    tool_calls = getattr(msg, "additional_kwargs", {}).get("tool_calls") or []
    parts = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function", {})
        parts.append({
            "type": "tool_call",
            "id": tc.get("id"),
            "name": func.get("name", ""),
            "arguments": _parse_arguments(func.get("arguments")),
        })
    return parts


def build_input_messages(messages: Any) -> List[Dict]:
    """Build OTel-compliant input messages from LlamaIndex ChatMessage list."""
    if not messages:
        return []
    result = []
    for msg in messages:
        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        parts = _content_to_parts(msg.content)

        if role == "assistant":
            parts.extend(_extract_tool_calls(msg))

        if role == "tool":
            parts = _maybe_wrap_tool_response(msg, parts)

        result.append({"role": role, "parts": parts})
    return result


def _maybe_wrap_tool_response(msg: Any, parts: List[Dict]) -> List[Dict]:
    """Wrap content as tool_call_response for tool-role messages if tool_call_id present."""
    tool_call_id = getattr(msg, "additional_kwargs", {}).get("tool_call_id")
    if not tool_call_id or not parts:
        return parts
    response_content = parts[0].get("content", "") if parts else ""
    return [{"type": "tool_call_response", "id": tool_call_id, "response": response_content}]


def build_output_message(response_message: Any, finish_reason: Optional[str] = None) -> Dict:
    """Build a single OTel-compliant output message from a LlamaIndex response message."""
    role = response_message.role.value if hasattr(response_message.role, "value") else "assistant"
    parts = _content_to_parts(response_message.content)
    parts.extend(_extract_tool_calls(response_message))
    fr = map_finish_reason(finish_reason) or ""
    return {"role": role, "parts": parts, "finish_reason": fr}


def build_completion_output_message(text: str, finish_reason: Optional[str] = None) -> Dict:
    """Build output message for text completion responses."""
    fr = map_finish_reason(finish_reason) or ""
    parts = [{"type": "text", "content": text}] if text else []
    return {"role": "assistant", "parts": parts, "finish_reason": fr}
