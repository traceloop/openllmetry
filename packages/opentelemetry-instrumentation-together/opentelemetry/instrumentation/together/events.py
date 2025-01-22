"""Event utilities for Together AI instrumentation."""

from typing import Any, Dict, Optional

from opentelemetry._events import Event
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

def message_to_event(
    message: Dict[str, Any],
    capture_content: bool = True,
    trace_id: Optional[int] = None,
    span_id: Optional[int] = None,
    trace_flags: Optional[int] = None,
) -> Event:
    """Create an event for a Together AI message."""
    event_attributes = {
        GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.TOGETHER.value,
    }

    body = {}
    if capture_content:
        body["role"] = message.get("role", "user")
        body["content"] = message.get("content")

    return Event(
        name="gen_ai.prompt",
        attributes=event_attributes,
        body=body,
        trace_id=trace_id,
        span_id=span_id,
        trace_flags=trace_flags,
    )

def choice_to_event(
    choice: Dict[str, Any],
    capture_content: bool = True,
    trace_id: Optional[int] = None,
    span_id: Optional[int] = None,
    trace_flags: Optional[int] = None,
) -> Event:
    """Create an event for a Together AI completion choice."""
    event_attributes = {
        GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.TOGETHER.value,
    }

    body = {}
    if capture_content:
        if "message" in choice:
            body["role"] = choice["message"].get("role", "assistant")
            body["content"] = choice["message"].get("content")
        else:
            body["role"] = "assistant"
            body["content"] = choice.get("text")
        body["finish_reason"] = choice.get("finish_reason")
        body["index"] = choice.get("index", 0)

    return Event(
        name="gen_ai.completion",
        attributes=event_attributes,
        body=body,
        trace_id=trace_id,
        span_id=span_id,
        trace_flags=trace_flags,
    )