"""Event utilities for LlamaIndex instrumentation."""

from typing import Any, Dict, Optional

from opentelemetry._events import Event
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes

def create_prompt_event(
    prompt: Dict[str, Any],
    capture_content: bool = True,
    trace_id: Optional[int] = None,
    span_id: Optional[int] = None,
    trace_flags: Optional[int] = None,
) -> Event:
    """Create an event for a LlamaIndex prompt."""
    event_attributes = {
        GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.LLAMAINDEX.value
    }

    body = {}
    if capture_content:
        body["prompt"] = prompt

    return Event(
        name="gen_ai.prompt",
        attributes=event_attributes,
        body=body,
        trace_id=trace_id,
        span_id=span_id,
        trace_flags=trace_flags,
    )

def create_completion_event(
    completion: Dict[str, Any],
    capture_content: bool = True,
    trace_id: Optional[int] = None,
    span_id: Optional[int] = None,
    trace_flags: Optional[int] = None,
) -> Event:
    """Create an event for a LlamaIndex completion."""
    event_attributes = {
        GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.LLAMAINDEX.value
    }

    body = {}
    if capture_content:
        body["completion"] = completion

    return Event(
        name="gen_ai.completion",
        attributes=event_attributes,
        body=body,
        trace_id=trace_id,
        span_id=span_id,
        trace_flags=trace_flags,
    )

def create_tool_call_event(
    tool_call: Dict[str, Any],
    capture_content: bool = True,
    trace_id: Optional[int] = None,
    span_id: Optional[int] = None,
    trace_flags: Optional[int] = None,
) -> Event:
    """Create an event for a LlamaIndex tool call."""
    event_attributes = {
        GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.LLAMAINDEX.value
    }

    body = {}
    if capture_content:
        body["tool_call"] = tool_call

    return Event(
        name="gen_ai.tool_call",
        attributes=event_attributes,
        body=body,
        trace_id=trace_id,
        span_id=span_id,
        trace_flags=trace_flags,
    ) 