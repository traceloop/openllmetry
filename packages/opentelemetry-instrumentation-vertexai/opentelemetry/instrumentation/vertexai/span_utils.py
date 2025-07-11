from opentelemetry.instrumentation.vertexai.utils import dont_throw, should_send_prompts
from opentelemetry.semconv_ai import SpanAttributes


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


@dont_throw
def set_input_attributes(span, args):
    if not span.is_recording():
        return
    if should_send_prompts() and args is not None and len(args) > 0:
        prompt = ""
        for arg in args:
            if isinstance(arg, str):
                prompt = f"{prompt}{arg}\n"
            elif isinstance(arg, list):
                for subarg in arg:
                    prompt = f"{prompt}{subarg}\n"

        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_PROMPTS}.0.user",
            prompt,
        )


@dont_throw
def set_model_input_attributes(span, kwargs, llm_model):
    if not span.is_recording():
        return
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, llm_model)
    _set_span_attribute(
        span, f"{SpanAttributes.LLM_PROMPTS}.0.user", kwargs.get("prompt")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TEMPERATURE, kwargs.get("temperature")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_output_tokens")
    )
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TOP_P, kwargs.get("top_p"))
    _set_span_attribute(span, SpanAttributes.LLM_TOP_K, kwargs.get("top_k"))
    _set_span_attribute(
        span, SpanAttributes.LLM_PRESENCE_PENALTY, kwargs.get("presence_penalty")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_FREQUENCY_PENALTY, kwargs.get("frequency_penalty")
    )


@dont_throw
def set_response_attributes(span, llm_model, generation_text):
    if not span.is_recording() or not should_send_prompts():
        return
    _set_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
    _set_span_attribute(
        span,
        f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
        generation_text,
    )


@dont_throw
def set_model_response_attributes(span, llm_model, token_usage):
    if not span.is_recording():
        return
    _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, llm_model)

    if token_usage:
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            token_usage.total_token_count,
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            token_usage.candidates_token_count,
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
            token_usage.prompt_token_count,
        )
