"""Event emission helpers for Bedrock instrumentation."""

import json
from typing import Any, Dict, List, Optional, Union

from opentelemetry.semconv_ai import SpanAttributes


def emit_prompt_event(
    span,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    index: int = 0,
) -> None:
    """Emit a prompt event for a Bedrock request.

    Handles both chat-style (messages) and completion-style (prompt) inputs.
    """
    event_attributes = {
        SpanAttributes.LLM_PROMPT_INDEX: index,
    }

    if messages:
        # Chat-style input (Claude, Anthropic, etc.)
        event_attributes[SpanAttributes.LLM_PROMPT_TYPE] = "chat"
        for i, msg in enumerate(messages):
            if msg.get("role"):
                event_attributes[f"messages.{i}.role"] = msg["role"]
            if msg.get("content"):
                content = msg["content"]
                if isinstance(content, list):  # Handle multi-modal content
                    content = json.dumps(content)
                event_attributes[f"messages.{i}.content"] = content
    else:
        # Completion-style input (Titan, etc.)
        event_attributes[SpanAttributes.LLM_PROMPT_TYPE] = "completion"
        if prompt:
            event_attributes[SpanAttributes.LLM_PROMPT_CONTENT] = prompt

    span.add_event(name="prompt", attributes=event_attributes)


def emit_completion_event(
    span,
    completion: Dict[str, Any],
    index: Optional[int] = None,
    is_streaming: bool = False,
) -> None:
    """Emit a completion event for a Bedrock response."""
    event_attributes = {}

    if index is not None:
        event_attributes[SpanAttributes.LLM_COMPLETION_INDEX] = index

    # Handle different model response formats
    if "completion" in completion:
        # Titan model format
        event_attributes[SpanAttributes.LLM_COMPLETION_CONTENT] = completion[
            "completion"
        ]
    elif "content" in completion:
        # Claude/Anthropic format
        content = completion["content"]
        if isinstance(content, list):  # Handle multi-modal content
            content = json.dumps(content)
        event_attributes[SpanAttributes.LLM_COMPLETION_CONTENT] = content
    elif "generation" in completion:
        # Cohere format
        event_attributes[SpanAttributes.LLM_COMPLETION_CONTENT] = completion[
            "generation"
        ]
    elif "text" in completion:
        # AI21 format
        event_attributes[SpanAttributes.LLM_COMPLETION_CONTENT] = completion["text"]

    if "stop_reason" in completion:
        event_attributes[SpanAttributes.LLM_RESPONSE_FINISH_REASON] = completion[
            "stop_reason"
        ]
    elif "finish_reason" in completion:
        event_attributes[SpanAttributes.LLM_RESPONSE_FINISH_REASON] = completion[
            "finish_reason"
        ]

    if "role" in completion:
        event_attributes[SpanAttributes.LLM_COMPLETION_ROLE] = completion["role"]

    # Handle tool calls
    if "tool_calls" in completion:
        tool_calls = completion["tool_calls"]
        for i, tool_call in enumerate(tool_calls):
            event_attributes[f"tool_calls.{i}.type"] = tool_call.get("type")
            if tool_call.get("function"):
                function = tool_call["function"]
                event_attributes[f"tool_calls.{i}.function.name"] = function.get("name")
                event_attributes[f"tool_calls.{i}.function.arguments"] = function.get(
                    "arguments"
                )

    event_name = (
        f"{SpanAttributes.LLM_CONTENT_COMPLETION_CHUNK}"
        if is_streaming
        else "completion"
    )
    span.add_event(name=event_name, attributes=event_attributes)
