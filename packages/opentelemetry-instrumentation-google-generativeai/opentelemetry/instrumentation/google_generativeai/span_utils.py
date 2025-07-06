from opentelemetry.instrumentation.google_generativeai.utils import (
    dont_throw,
    should_send_prompts,
)
from opentelemetry.semconv_ai import (
    SpanAttributes,
)
from opentelemetry.trace.status import Status, StatusCode


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


@dont_throw
def set_input_attributes(span, args, kwargs, llm_model):
    if not span.is_recording():
        return

    if not should_send_prompts():
        return

    if "contents" in kwargs:
        contents = kwargs["contents"]
        if isinstance(contents, str):
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.0.content",
                contents,
            )
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.0.role",
                "user",
            )
        elif isinstance(contents, list):
            for i, content in enumerate(contents):
                if hasattr(content, "parts"):
                    for part in content.parts:
                        if hasattr(part, "text"):
                            _set_span_attribute(
                                span,
                                f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                                part.text,
                            )
                            _set_span_attribute(
                                span,
                                f"{SpanAttributes.LLM_PROMPTS}.{i}.role",
                                getattr(content, "role", "user"),
                            )
    elif args and len(args) > 0:
        prompt = ""
        for arg in args:
            if isinstance(arg, str):
                prompt = f"{prompt}{arg}\n"
            elif isinstance(arg, list):
                for subarg in arg:
                    prompt = f"{prompt}{subarg}\n"
        if prompt:
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.0.content",
                prompt,
            )
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.0.role",
                "user",
            )
    elif "prompt" in kwargs:
        _set_span_attribute(
            span, f"{SpanAttributes.LLM_PROMPTS}.0.content", kwargs["prompt"]
        )
        _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")


def set_model_request_attributes(span, kwargs, llm_model):
    if not span.is_recording():
        return
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, llm_model)
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
def set_response_attributes(span, response, llm_model):
    if not should_send_prompts():
        return
    if hasattr(response, "usage_metadata"):
        if isinstance(response.text, list):
            for index, item in enumerate(response):
                prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
                _set_span_attribute(span, f"{prefix}.content", item.text)
                _set_span_attribute(span, f"{prefix}.role", "assistant")
        elif isinstance(response.text, str):
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", response.text
            )
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant"
            )
    else:
        if isinstance(response, list):
            for index, item in enumerate(response):
                prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
                _set_span_attribute(span, f"{prefix}.content", item)
                _set_span_attribute(span, f"{prefix}.role", "assistant")
        elif isinstance(response, str):
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", response
            )
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant"
            )


def set_model_response_attributes(span, response, llm_model):
    if not span.is_recording():
        return

    _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, llm_model)

    if hasattr(response, "usage_metadata"):
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            response.usage_metadata.total_token_count,
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            response.usage_metadata.candidates_token_count,
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
            response.usage_metadata.prompt_token_count,
        )

    span.set_status(Status(StatusCode.OK))
