from contextlib import contextmanager
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry import context
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Span, set_span_in_context
from pydantic import BaseModel
from typing import Optional
from traceloop.sdk.tracing.context_manager import get_tracer


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


class LLMSpan:
    _span: Span = None

    def __init__(self, span: Span):
        self._span = span
        pass

    def report_request(self, model: str, messages: list[LLMMessage]):
        self._span.set_attribute(GenAIAttributes.GEN_AI_REQUEST_MODEL, model)
        for idx, message in enumerate(messages):
            self._span.set_attribute(
                f"{GenAIAttributes.GEN_AI_PROMPT}.{idx}.role", message.role
            )
            self._span.set_attribute(
                f"{GenAIAttributes.GEN_AI_PROMPT}.{idx}.content", message.content
            )

    def report_response(self, model: str, completions: list[str]):
        self._span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_MODEL, model)
        for idx, completion in enumerate(completions):
            self._span.set_attribute(
                f"{GenAIAttributes.GEN_AI_COMPLETION}.{idx}.role", "assistant"
            )
            self._span.set_attribute(
                f"{GenAIAttributes.GEN_AI_COMPLETION}.{idx}.content", completion
            )

    def report_usage(self, usage: LLMUsage):
        self._span.set_attribute(
            GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, usage.prompt_tokens
        )
        self._span.set_attribute(
            GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, usage.completion_tokens
        )
        self._span.set_attribute(
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.total_tokens
        )
        if usage.cache_creation_input_tokens is not None:
            self._span.set_attribute(
                SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
                usage.cache_creation_input_tokens,
            )
        if usage.cache_read_input_tokens is not None:
            self._span.set_attribute(
                SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
                usage.cache_read_input_tokens,
            )


@contextmanager
def track_llm_call(vendor: str, type: str):
    with get_tracer() as tracer:
        span = tracer.start_span(name=f"{vendor}.{type}")
        span.set_attribute(GenAIAttributes.GEN_AI_SYSTEM, vendor)
        span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, type)
        ctx = set_span_in_context(span)
        token = context.attach(ctx)
        try:
            yield LLMSpan(span)
        finally:
            context.detach(token)
            span.end()
