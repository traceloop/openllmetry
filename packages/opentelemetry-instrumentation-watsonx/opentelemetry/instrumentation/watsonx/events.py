"""Event utilities for WatsonX instrumentation."""

from typing import Any, Dict, Optional

from opentelemetry._events import Event
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

def prompt_to_event(
    prompt: str,
    model_name: str,
    capture_content: bool = True,
    trace_id: Optional[int] = None,
    span_id: Optional[int] = None,
    trace_flags: Optional[int] = None,
) -> Event:
    """Create an event for a WatsonX prompt."""
    event_attributes = {
        GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.WATSONX.value,
        GenAIAttributes.GEN_AI_REQUEST_MODEL: model_name,
    }

    body = {}
    if capture_content:
        body["content"] = prompt

    return Event(
        name="gen_ai.prompt",
        attributes=event_attributes,
        body=body,
        trace_id=trace_id,
        span_id=span_id,
        trace_flags=trace_flags,
    )

def completion_to_event(
    completion: Dict[str, Any],
    model_name: str,
    capture_content: bool = True,
    trace_id: Optional[int] = None,
    span_id: Optional[int] = None,
    trace_flags: Optional[int] = None,
) -> Event:
    """Create an event for a WatsonX completion."""
    event_attributes = {
        GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.WATSONX.value,
        GenAIAttributes.GEN_AI_RESPONSE_MODEL: model_name,
    }

    body = {}
    if capture_content:
        if isinstance(completion, dict):
            body["content"] = completion.get("generated_text", "")
            if "token_usage" in completion:
                body["token_usage"] = completion["token_usage"]
        else:
            body["content"] = str(completion)

    return Event(
        name="gen_ai.completion",
        attributes=event_attributes,
        body=body,
        trace_id=trace_id,
        span_id=span_id,
        trace_flags=trace_flags,
    ) 