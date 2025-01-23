"""OpenTelemetry Replicate Events Utilities"""

from opentelemetry._events import Event
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

def prompt_to_event(prompt, model: str, capture_content: bool = True) -> Event:
    """Convert a prompt to an event."""
    body = {
        "model": model,
    }
    
    if capture_content and prompt is not None:
        if isinstance(prompt, (list, dict)):
            body["prompt"] = prompt
        else:
            body["prompt"] = str(prompt)

    event_attributes = {
        GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.REPLICATE.value
    }

    return Event(
        name="gen_ai.prompt",
        attributes=event_attributes,
        body=body,
    )

def completion_to_event(completion, model: str, capture_content: bool = True) -> Event:
    """Convert a completion response to an event."""
    body = {
        "model": model,
    }

    if capture_content and completion is not None:
        if isinstance(completion, (list, dict)):
            body["completion"] = completion
        else:
            body["completion"] = str(completion)

    event_attributes = {
        GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.REPLICATE.value
    }

    return Event(
        name="gen_ai.completion",
        attributes=event_attributes,
        body=body,
    ) 