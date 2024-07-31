from contextlib import contextmanager
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Span
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
        self._span = span
        pass

    def report_request(self, model: str, messages: list[LLMMessage]):
        self._span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model)
        for idx, message in enumerate(messages):
            self._span.set_attribute(
                f"{SpanAttributes.LLM_PROMPTS}.{idx}.role", message.role
            )
            self._span.set_attribute(
                f"{SpanAttributes.LLM_PROMPTS}.{idx}.content", message.content
            )

    def report_response(self, model: str, completions: list[str]):
        self._span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, model)
        for idx, completion in enumerate(completions):
            self._span.set_attribute(
                f"{SpanAttributes.LLM_COMPLETIONS}.{idx}.role", "assistant"
            )
            self._span.set_attribute(
                f"{SpanAttributes.LLM_COMPLETIONS}.{idx}", completion
            )


@contextmanager
def track_llm_call(vendor: str, type: str):
    with get_tracer() as tracer:
        with tracer.start_as_current_span(name=f"{vendor}.{type}") as span:
            span.set_attribute(SpanAttributes.LLM_SYSTEM, vendor)
            span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, type)
            llm_span = LLMSpan(span)
            try:
                yield llm_span
            finally:
                span.end()
