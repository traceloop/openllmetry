"""OpenTelemetry Ollama Events Utilities"""

from opentelemetry._events import Event
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

def message_to_event(message, capture_content: bool = True) -> Event:
    """Convert a chat message to an event."""
    body = {
        "role": message.get("role"),
    }
    
    if capture_content and message.get("content") is not None:
        body["content"] = message["content"]

    event_attributes = {
        GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.OLLAMA.value
    }

    return Event(
        name="gen_ai.prompt",
        attributes=event_attributes,
        body=body,
    )

def completion_to_event(completion, capture_content: bool = True) -> Event:
    """Convert a completion response to an event."""
    body = {
        "model": completion.get("model"),
        "created_at": completion.get("created_at"),
        "done": completion.get("done", False),
    }

    if capture_content:
        if completion.get("response") is not None:
            body["content"] = completion["response"]
        if completion.get("context") is not None:
            body["context"] = completion["context"]

    event_attributes = {
        GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.OLLAMA.value
    }

    return Event(
        name="gen_ai.completion",
        attributes=event_attributes,
        body=body,
    )

def embedding_to_event(embedding, input_text: str, capture_content: bool = True) -> Event:
    """Convert an embedding response to an event."""
    body = {
        "model": embedding.get("model"),
        "embedding": embedding.get("embedding"),
    }

    if capture_content:
        body["input"] = input_text

    event_attributes = {
        GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.OLLAMA.value
    }

    return Event(
        name="gen_ai.embedding",
        attributes=event_attributes,
        body=body,
    ) 