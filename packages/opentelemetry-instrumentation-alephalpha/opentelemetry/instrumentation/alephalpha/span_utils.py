from opentelemetry.instrumentation.alephalpha.event_models import (
    CompletionEvent,
    PromptEvent,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace.span import Span


def set_prompt_attributes(event: PromptEvent, span: Span):
    from opentelemetry.instrumentation.alephalpha import (
        _set_span_attribute,
        should_send_prompts,
    )

    if not span.is_recording():
        return

    if should_send_prompts():
        _set_span_attribute(span, f"{GenAIAttributes.GEN_AI_PROMPT}.0.role", "user")
        _set_span_attribute(
            span,
            f"{GenAIAttributes.GEN_AI_PROMPT}.0.content",
            event.content[0].get("data"),
        )


def set_completion_attributes(event: CompletionEvent, span: Span):
    from opentelemetry.instrumentation.alephalpha import (
        _set_span_attribute,
        should_send_prompts,
    )

    if not span.is_recording():
        return

    if should_send_prompts():
        _set_span_attribute(
            span,
            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content",
            event.message["content"],
        )
        _set_span_attribute(
            span, f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role", "assistant"
        )
