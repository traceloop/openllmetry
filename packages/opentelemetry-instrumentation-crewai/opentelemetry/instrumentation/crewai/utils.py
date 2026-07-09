import json


def _messages_to_otel_input(messages) -> str | None:
    """Convert CrewAI input messages to OTel gen_ai.input.messages JSON."""
    if messages is None:
        return None
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not isinstance(messages, list):
        return None

    result = [_convert_message(msg) for msg in messages]
    result = [m for m in result if m is not None]
    return json.dumps(result) if result else None


def _convert_message(msg) -> dict | None:
    role = msg.get("role", "user")

    if role == "tool":
        return _convert_tool_result(msg)

    parts = _extract_content_parts(msg.get("content"))
    parts += _extract_tool_call_parts(msg.get("tool_calls"))

    return {"role": role, "parts": parts} if parts else None


def _convert_tool_result(msg) -> dict:
    content = msg.get("content")
    return {
        "role": "tool",
        "parts": [{
            "type": "tool_call_response",
            "id": msg.get("tool_call_id", ""),
            "response": str(content) if content is not None else "",
        }],
    }


def _extract_content_parts(content) -> list:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "content": content}]
    if isinstance(content, list):
        return [_convert_content_block(b) for b in content if isinstance(b, dict)]
    return [{"type": "text", "content": str(content)}]


def _convert_content_block(block) -> dict:
    btype = block.get("type", "")

    if btype == "text":
        return {"type": "text", "content": block.get("text", "")}

    if btype == "image_url":
        url = (block.get("image_url") or {}).get("url", "")
        return {"type": "uri", "modality": "image", "uri": url}

    # Unknown block — narrow GenericPart, don't leak raw provider fields
    generic = {"type": block.get("type", "unknown")}
    if "text" in block:
        generic["content"] = str(block["text"])
    elif "content" in block:
        generic["content"] = str(block["content"])
    return generic


def _extract_tool_call_parts(tool_calls) -> list:
    if not tool_calls or not isinstance(tool_calls, list):
        return []
    return [_convert_tool_call(tc) for tc in tool_calls]


def _convert_tool_call(tc) -> dict:
    fn = tc.get("function") or {}
    raw_args = fn.get("arguments", "{}")
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except (json.JSONDecodeError, TypeError):
        args = raw_args
    return {"type": "tool_call", "id": tc.get("id", ""), "name": fn.get("name", ""), "arguments": args}


def _response_to_otel_output(response) -> str | None:
    """Convert CrewAI LLM text response to OTel gen_ai.output.messages JSON."""
    if response is None:
        return None
    text = str(response)
    if not text:
        return None
    return json.dumps([{
        "role": "assistant",
        "parts": [{"type": "text", "content": text}],
    }])
