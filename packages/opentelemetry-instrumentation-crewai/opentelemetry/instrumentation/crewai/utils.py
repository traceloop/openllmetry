import json
import os
from opentelemetry import context as context_api

TRACELOOP_TRACE_CONTENT = "TRACELOOP_TRACE_CONTENT"


def should_send_prompts() -> bool:
    return (
        os.getenv(TRACELOOP_TRACE_CONTENT) or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def _messages_to_otel_input(messages) -> str | None:
    """Convert CrewAI input messages (str or list[dict]) to OTel gen_ai.input.messages JSON."""
    if messages is None:
        return None
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not isinstance(messages, list):
        return None

    result = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        parts = []

        # Tool result messages (role: "tool") → tool_call_response part
        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            response_text = str(content) if content is not None else ""
            parts.append({"type": "tool_call_response", "id": tool_call_id, "response": response_text})
            result.append({"role": role, "parts": parts})
            continue

        # Build content parts from the content field
        if content is not None:
            if isinstance(content, str):
                parts = [{"type": "text", "content": content}]
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type", "")
                        if btype == "text":
                            parts.append({"type": "text", "content": block.get("text", "")})
                        elif btype == "image_url":
                            url = (block.get("image_url") or {}).get("url", "")
                            parts.append({"type": "uri", "modality": "image", "uri": url})
                        else:
                            parts.append({"type": block.get("type", "unknown"),
                                          **{k: v for k, v in block.items() if k != "type"}})
            else:
                parts = [{"type": "text", "content": str(content)}]

        # Assistant tool_calls → tool_call parts
        tool_calls = msg.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            for tc in tool_calls:
                fn = tc.get("function") or {}
                raw_args = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except (json.JSONDecodeError, TypeError):
                    args = raw_args
                parts.append({"type": "tool_call", "id": tc.get("id", ""),
                              "name": fn.get("name", ""), "arguments": args})

        if parts:
            result.append({"role": role, "parts": parts})

    return json.dumps(result) if result else None


def _response_to_otel_output(response) -> str | None:
    """Convert CrewAI LLM text response to OTel gen_ai.output.messages JSON."""
    if response is None:
        return None
    text = str(response)
    if not text:
        return None
    return json.dumps([
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": text}],
        }
    ])
