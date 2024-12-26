from contextlib import contextmanager
from opentelemetry.trace import Span, Status, StatusCode, set_span_in_context
from opentelemetry.context import attach, detach
from pydantic import BaseModel
from traceloop.sdk.tracing.context_manager import get_tracer


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMSpan:
    _span: Span = None

    def __init__(self, span: Span):
        if not span:
            raise ValueError("Span cannot be None")
        self._span = span
        print(f"LLMSpan initialized with span: {span}")

    def report_request(self, model: str, messages: list[LLMMessage]):
        if not self._span:
            raise ValueError("Span not initialized")
        
        # Set model attribute
        self._span.set_attribute("gen_ai.request.model", model)
        
        # Set message attributes
        for idx, message in enumerate(messages):
            # Set attributes with explicit string keys
            self._span.set_attribute(f"gen_ai.prompt.{idx}.role", message.role)
            self._span.set_attribute(f"gen_ai.prompt.{idx}.content", message.content)
            self._span.set_attribute(f"gen_ai.prompt.{idx}.system", "openai")
            
            # Debug output
            print(f"Set attributes for message {idx}:")
            print(f"  Role: {message.role}")
            print(f"  Content: {message.content}")
            print(f"Current attributes: {self._span.attributes}")

    def report_response(self, model: str, completions: list[str]):
        if not self._span:
            raise ValueError("Span not initialized")
        
        # Set model attribute
        self._span.set_attribute("gen_ai.response.model", model)
        
        # Set completion attributes
        for idx, completion in enumerate(completions):
            # Set attributes with explicit string keys
            self._span.set_attribute(f"gen_ai.completion.{idx}.role", "assistant")
            self._span.set_attribute(f"gen_ai.completion.{idx}.content", completion)
            
            # Debug output
            print(f"Set completion {idx}:")
            print(f"  Content: {completion}")
            print(f"Current attributes: {self._span.attributes}")


@contextmanager
def track_llm_call(vendor: str, type: str):
    with get_tracer() as tracer:
        # Use start_as_current_span to automatically manage context
        with tracer.start_as_current_span(name=f"{vendor}.{type}") as span:
            # Set base attributes
            span.set_attribute("gen_ai.system", vendor)
            span.set_attribute("gen_ai.operation.name", type)
            span.set_attribute("llm.request.type", type)  # Add this as it appears in test output
            
            # Create LLMSpan and yield it
            llm_span = LLMSpan(span)
            try:
                yield llm_span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
