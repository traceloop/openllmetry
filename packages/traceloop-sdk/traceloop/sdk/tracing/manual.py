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
        if not self._span or not self._span.is_recording():
            raise ValueError("Span not initialized or not recording")
        
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
        if not self._span or not self._span.is_recording():
            raise ValueError("Span not initialized or not recording")
        
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
        # Create a new span
        span = tracer.start_span(name=f"{vendor}.{type}")
        
        # Set base attributes
        span.set_attribute("gen_ai.system", vendor)
        span.set_attribute("gen_ai.operation.name", type)
        span.set_attribute("llm.request.type", type)
        
        # Attach context and create LLMSpan
        token = attach(set_span_in_context(span))
        llm_span = LLMSpan(span)
        
        try:
            yield llm_span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            # Always detach context and end span
            detach(token)
            span.end()
