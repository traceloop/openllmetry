from opentelemetry.instrumentation.together.utils import dont_throw, should_send_prompts
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import (
    LLMRequestTypeValues,
    SpanAttributes,
)


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


@dont_throw
def set_prompt_attributes(span, llm_request_type, kwargs):
    if not span.is_recording():
        return

    if should_send_prompts():
        if llm_request_type == LLMRequestTypeValues.CHAT:
            _set_span_attribute(span, f"{GenAIAttributes.GEN_AI_PROMPT}.0.role", "user")
            for index, message in enumerate(kwargs.get("messages")):
                _set_span_attribute(
                    span,
                    f"{GenAIAttributes.GEN_AI_PROMPT}.{index}.content",
                    message.get("content"),
                )
                _set_span_attribute(
                    span,
                    f"{GenAIAttributes.GEN_AI_PROMPT}.{index}.role",
                    message.get("role"),
                )
        elif llm_request_type == LLMRequestTypeValues.COMPLETION:
            _set_span_attribute(span, f"{GenAIAttributes.GEN_AI_PROMPT}.0.role", "user")
            _set_span_attribute(
                span, f"{GenAIAttributes.GEN_AI_PROMPT}.0.content", kwargs.get("prompt")
            )


@dont_throw
def set_model_prompt_attributes(span, kwargs):
    if not span.is_recording():
        return

    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MODEL, kwargs.get("model"))
    _set_span_attribute(
        span,
        SpanAttributes.LLM_IS_STREAMING,
        kwargs.get("stream"),
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS, kwargs.get("max_tokens")
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, kwargs.get("temperature")
    )
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_P, kwargs.get("top_p"))
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_K, kwargs.get("top_k"))
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY,
        kwargs.get("frequency_penalty"),
    )
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY,
        kwargs.get("presence_penalty"),
    )
    _set_span_attribute(
        span,
        SpanAttributes.GEN_AI_REQUEST_REPETITION_PENALTY,
        kwargs.get("repetition_penalty"),
    )
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_SEED, kwargs.get("seed"))
    stop = kwargs.get("stop")
    if isinstance(stop, str):
        stop = [stop] if stop else None
    if stop:
        _set_span_attribute(
            span, GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES, list(stop)
        )


@dont_throw
def set_completion_attributes(span, llm_request_type, response):
    if not span.is_recording():
        return

    if should_send_prompts():
        if llm_request_type == LLMRequestTypeValues.COMPLETION:
            _set_span_attribute(
                span,
                f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content",
                response.choices[0].text,
            )
            _set_span_attribute(
                span, f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role", "assistant"
            )
        elif llm_request_type == LLMRequestTypeValues.CHAT:
            index = 0
            prefix = f"{GenAIAttributes.GEN_AI_COMPLETION}.{index}"
            _set_span_attribute(
                span, f"{prefix}.content", response.choices[0].message.content
            )
            _set_span_attribute(
                span, f"{prefix}.role", response.choices[0].message.role
            )


@dont_throw
def set_model_completion_attributes(span, response):
    if not span.is_recording():
        return

    _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, response.model)
    _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, response.id)

    usage_data = response.usage
    input_tokens = getattr(usage_data, "prompt_tokens", 0)
    output_tokens = getattr(usage_data, "completion_tokens", 0)

    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
        input_tokens + output_tokens,
    )
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
        output_tokens,
    )
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
        input_tokens,
    )
