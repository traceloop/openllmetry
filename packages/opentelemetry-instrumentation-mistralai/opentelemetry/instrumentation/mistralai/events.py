"""OpenTelemetry Mistral AI Events Utilities"""

from opentelemetry._events import Event
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

def message_to_event(message, capture_content: bool = True) -> Event:
    """Convert a chat message to an event."""
    body = {
        "role": message.role,
    }
    
    if capture_content and message.content is not None:
        body["content"] = message.content

    event_attributes = {
        GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.MISTRAL.value
    }

    return Event(
        name="gen_ai.prompt",
        attributes=event_attributes,
        body=body,
    )

def choice_to_event(choice, capture_content: bool = True) -> Event:
    """Convert a completion choice to an event."""
    body = {
        "index": choice.index,
        "finish_reason": choice.finish_reason or "error",
        "message": {
            "role": choice.message.role,
        }
    }

    if capture_content and choice.message.content is not None:
        body["message"]["content"] = choice.message.content

    event_attributes = {
        GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.MISTRAL.value
    }

    return Event(
        name="gen_ai.choice",
        attributes=event_attributes,
        body=body,
    )

def embedding_to_event(embedding, input_text: str, capture_content: bool = True) -> Event:
    """Convert an embedding response to an event."""
    body = {
        "index": embedding.index,
        "embedding": embedding.embedding,
    }

    if capture_content:
        body["input"] = input_text

    event_attributes = {
        GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.MISTRAL.value
    }

    return Event(
        name="gen_ai.embedding",
        attributes=event_attributes,
        body=body,
    ) 