import logging
from typing import Optional, Dict, Any
from opentelemetry.instrumentation.alephalpha.config import Config
import traceback
from opentelemetry._events import Event
from opentelemetry.trace import get_current_span, Status, StatusCode
from opentelemetry.semconv_ai import (
    SpanAttributes,
    LLMRequestTypeValues,
)

def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.
    """
    logger = logging.getLogger(func.__module__)

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.debug(
                "OpenLLMetry failed to trace in %s, error: %s",
                func.__name__,
                traceback.format_exc(),
            )
            if Config.exception_logger:
                Config.exception_logger(e)

    return wrapper

def get_llm_request_attributes(kwargs: Dict[str, Any], instance: Any = None) -> Dict[str, Any]:
    """Get common LLM request attributes."""
    attributes = {
        SpanAttributes.LLM_SYSTEM: "AlephAlpha",
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
    }
    
    if "model" in kwargs:
        attributes[SpanAttributes.GEN_AI_REQUEST_MODEL] = kwargs["model"]
    
    return attributes

def message_to_event(prompt_text: str, capture_content: bool = True) -> Event:
    """Convert a prompt message to an event."""
    attributes = {
        SpanAttributes.LLM_SYSTEM: "AlephAlpha",
    }
    
    body = {
        "role": "user",
    }
    
    if capture_content:
        body["content"] = prompt_text
        
    # Get current span context for trace propagation
    span = get_current_span()
    span_context = span.get_span_context()
    
    return Event(
        name="gen_ai.prompt",
        attributes=attributes,
        body=body,
        trace_id=span_context.trace_id,
        span_id=span_context.span_id,
        trace_flags=span_context.trace_flags,
    )

def completion_to_event(completion_text: str, capture_content: bool = True) -> Event:
    """Convert a completion to an event."""
    attributes = {
        SpanAttributes.LLM_SYSTEM: "AlephAlpha",
    }
    
    body = {
        "role": "assistant",
    }
    
    if capture_content:
        body["content"] = completion_text
        
    # Get current span context for trace propagation
    span = get_current_span()
    span_context = span.get_span_context()
    
    return Event(
        name="gen_ai.completion",
        attributes=attributes,
        body=body,
        trace_id=span_context.trace_id,
        span_id=span_context.span_id,
        trace_flags=span_context.trace_flags,
    )

def set_span_attribute(span, name: str, value: Any):
    """Set span attribute if value is not None and not empty."""
    if value is not None and value != "":
        span.set_attribute(name, value)

def handle_span_exception(span, error: Exception):
    """Handle span exception by recording error and ending span."""
    if span.is_recording():
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR))
    span.end()

class CompletionBuffer:
    """Buffer for streaming completions."""
    def __init__(self, index: int):
        self.index = index
        self.text_content = []
        self.finish_reason = None

    def append_content(self, content: str):
        self.text_content.append(content)

    def get_content(self) -> str:
        return "".join(self.text_content)
