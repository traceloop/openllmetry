"""Event utilities for Anthropic instrumentation."""

import json
from typing import Any, Dict, Optional

from opentelemetry._events import Event
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
from opentelemetry.trace import SpanContext

def create_base_event(name: str, span_ctx: SpanContext) -> Event:
    """Creates a base event with proper trace context."""
    return Event(
        name=name,
        attributes={"gen_ai.system": "anthropic"},
        trace_id=span_ctx.trace_id,
        span_id=span_ctx.span_id,
        trace_flags=span_ctx.trace_flags
    )

def create_prompt_event(
    content: str,
    role: str,
    span_ctx: SpanContext,
    system: Optional[str] = None,
    index: int = 0,
) -> Event:
    """Creates an event for a prompt message."""
    event = create_base_event("gen_ai.prompt", span_ctx)
    
    message = {
        "role": role,
        "content": content,
        "index": index,
    }
    
    if system:
        message["system"] = system
        
    event.body = message
    return event

def create_completion_event(
    content: str,
    role: str,
    finish_reason: Optional[str],
    span_ctx: SpanContext,
    index: int = 0,
    tool_calls: Optional[list] = None,
) -> Event:
    """Creates an event for a completion message."""
    event = create_base_event("gen_ai.completion", span_ctx)
    
    message = {
        "role": role,
        "content": content,
        "index": index,
        "finish_reason": finish_reason
    }
    
    if tool_calls:
        message["tool_calls"] = tool_calls
        
    event.body = message
    return event

def create_tool_call_event(
    tool_call: Dict[str, Any],
    span_ctx: SpanContext,
    index: int = 0,
) -> Event:
    """Creates an event for a tool call."""
    event = create_base_event("gen_ai.tool_call", span_ctx)
    
    tool_call["index"] = index
    event.body = tool_call
    return event 