from opentelemetry.semconv_ai import SpanAttributes

from opentelemetry.instrumentation.writer.utils import (
    dont_throw,
    set_span_attribute,
    should_send_prompts,
    model_as_dict,
)


@dont_throw
def set_input_attributes(span, kwargs):
    if not span.is_recording():
        return

    if should_send_prompts():
        if kwargs.get("prompt") is not None:
            set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
            set_span_attribute(
                span, f"{SpanAttributes.LLM_PROMPTS}.0.content", kwargs.get("prompt")
            )

        elif kwargs.get("messages") is not None:
            for i, message in enumerate(kwargs.get("messages")):
                set_span_attribute(
                    span, f"{SpanAttributes.LLM_PROMPTS}.{i}.role", message.get("role")
                )
                set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                    message.get("content"),
                )
            # TODO add tool calls setter


@dont_throw
def set_model_input_attributes(span, kwargs):
    if not span.is_recording():
        return

    set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
    set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_tokens")
    )
    set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TEMPERATURE, kwargs.get("temperature")
    )
    set_span_attribute(span, SpanAttributes.LLM_REQUEST_TOP_P, kwargs.get("top_p"))
    set_span_attribute(span, SpanAttributes.LLM_CHAT_STOP_SEQUENCES, kwargs.get("stop"))
    set_span_attribute(
        span, SpanAttributes.LLM_IS_STREAMING, kwargs.get("stream") or False
    )


@dont_throw
def set_model_response_attributes(span, response, token_histogram):
    if not span.is_recording():
        return
    
    response_dict = model_as_dict(response)
    
    set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, response_dict.get("model"))
    
    usage = response_dict.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    
    if usage:
        set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)
        set_span_attribute(span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens)
        set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
    
    if (
        isinstance(prompt_tokens, int)
        and prompt_tokens >= 0
        and token_histogram is not None
    ):
        token_histogram.record(
            prompt_tokens,
            attributes={
                SpanAttributes.LLM_TOKEN_TYPE: "input",
                SpanAttributes.LLM_RESPONSE_MODEL: response_dict.get("model"),
            },
        )

    if (
        isinstance(completion_tokens, int)
        and completion_tokens >= 0
        and token_histogram is not None
    ):
        token_histogram.record(
            completion_tokens,
            attributes={
                SpanAttributes.LLM_TOKEN_TYPE: "output",
                SpanAttributes.LLM_RESPONSE_MODEL: response_dict.get("model"),
            },
        )


@dont_throw
def set_response_attributes(span, response):
    if not span.is_recording():
        return
    
    response_dict = model_as_dict(response)
    choices = response_dict.get("choices")
    
    if should_send_prompts() and choices:
        _set_completions(span, choices)


def _set_completions(span, choices):
    if choices is None or not should_send_prompts():
        return

    for choice in choices:
        index = choice.get("index", 0)
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
        set_span_attribute(span, f"{prefix}.finish_reason", choice.get("finish_reason"))
        
        message = choice.get("message")
        if message:
            set_span_attribute(span, f"{prefix}.role", message.get("role", "assistant"))
            set_span_attribute(span, f"{prefix}.content", message.get("content"))
            
            # Handle tool calls if present
            tool_calls = message.get("tool_calls")
            if tool_calls:
                for i, tool_call in enumerate(tool_calls):
                    function = tool_call.get("function", {})
                    set_span_attribute(
                        span, f"{prefix}.tool_calls.{i}.id", tool_call.get("id")
                    )
                    set_span_attribute(
                        span, f"{prefix}.tool_calls.{i}.name", function.get("name")
                    )
                    set_span_attribute(
                        span, f"{prefix}.tool_calls.{i}.arguments", function.get("arguments")
                    )
        
        elif choice.get("text") is not None:
            set_span_attribute(span, f"{prefix}.role", "assistant")
            set_span_attribute(span, f"{prefix}.content", choice.get("text"))


def set_streaming_response_attributes(span, accumulated_content, finish_reason=None, usage=None):
    if not span.is_recording() or not should_send_prompts():
        return

    prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0"
    set_span_attribute(span, f"{prefix}.role", "assistant")
    set_span_attribute(span, f"{prefix}.content", accumulated_content)
    if finish_reason:
        set_span_attribute(span, f"{prefix}.finish_reason", finish_reason)


def set_model_streaming_response_attributes(span, usage):
    if not span.is_recording():
        return

    if usage:
        set_span_attribute(
            span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, usage.get("completion_tokens")
        )
        set_span_attribute(
            span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.get("prompt_tokens")
        )
        set_span_attribute(
            span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.get("total_tokens")
        )
