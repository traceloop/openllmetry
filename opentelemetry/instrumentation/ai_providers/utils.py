from opentelemetry._events import Event, EventLogger
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes

def create_prompt_event(prompt: dict, system: str, capture_content: bool = True) -> Event:
    """Creates a standardized prompt event"""
    attributes = {
        GenAIAttributes.GEN_AI_SYSTEM: system
    }
    
    body = {
        "role": prompt.get("role", "user"),
    }
    
    if capture_content and "content" in prompt:
        body["content"] = prompt["content"]
        
    return Event(
        name="gen_ai.prompt",
        attributes=attributes,
        body=body
    )

def create_completion_event(completion: dict, system: str, capture_content: bool = True) -> Event:
    """Creates a standardized completion event"""
    # Similar implementation for completions
    pass 