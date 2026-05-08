import json


def set_span_attribute(span, name, value):
    if value is not None and value != "":
        span.set_attribute(name, value)


def messages_to_otel_input(messages) -> str | None:
    """Convert DSPy / litellm input messages to OTel gen_ai.input.messages JSON."""
    if messages is None:
        return None
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not isinstance(messages, list):
        return None

    result = [_convert_message(m) for m in messages]
    result = [m for m in result if m is not None]
    return json.dumps(result) if result else None


def response_to_otel_output(result) -> str | None:
    """Convert litellm ModelResponse to OTel gen_ai.output.messages JSON."""
    try:
        choices = result.get("choices") if hasattr(result, "get") else getattr(result, "choices", None)
        if not choices:
            return None
        out = []
        for choice in choices:
            msg = getattr(choice, "message", None)
            if msg is None and hasattr(choice, "get"):
                msg = choice.get("message")
            if msg is None:
                continue
            parts = _extract_content_parts(_get(msg, "content"))
            parts += _extract_tool_call_parts(_get(msg, "tool_calls"))
            if not parts:
                continue
            out.append({
                "role": _get(msg, "role") or "assistant",
                "parts": parts,
                "finish_reason": getattr(choice, "finish_reason", None) or _get(choice, "finish_reason"),
            })
        return json.dumps(out) if out else None
    except Exception:
        return None


def get_token_usage(result):
    try:
        usage = getattr(result, "usage", None)
        if usage is None and hasattr(result, "get"):
            usage = result.get("usage")
        if usage is None:
            return None, None
        prompt = _get(usage, "prompt_tokens")
        completion = _get(usage, "completion_tokens")
        return prompt, completion
    except Exception:
        return None, None


def _get(obj, key):
    if obj is None:
        return None
    val = getattr(obj, key, None)
    if val is None and hasattr(obj, "get"):
        try:
            val = obj.get(key)
        except Exception:
            val = None
    return val


def _convert_message(msg) -> dict | None:
    if not isinstance(msg, dict):
        return None
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
        return [{"type": "text", "content": content}] if content else []
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
    generic = {"type": btype or "unknown"}
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
    if not isinstance(tc, dict):
        tc = {
            "id": getattr(tc, "id", ""),
            "function": getattr(tc, "function", None),
        }
    fn = tc.get("function") or {}
    if not isinstance(fn, dict):
        fn = {"name": getattr(fn, "name", ""), "arguments": getattr(fn, "arguments", "{}")}
    raw_args = fn.get("arguments", "{}")
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except (json.JSONDecodeError, TypeError):
        args = raw_args
    return {"type": "tool_call", "id": tc.get("id", ""), "name": fn.get("name", ""), "arguments": args}
