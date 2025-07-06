import json

from opentelemetry.instrumentation.groq.utils import (
    dont_throw,
    model_as_dict,
    set_span_attribute,
    should_send_prompts,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_RESPONSE_ID,
)
from opentelemetry.semconv_ai import (
    SpanAttributes,
)

CONTENT_FILTER_KEY = "content_filter_results"


@dont_throw
def set_input_attributes(span, kwargs):
    if not span.is_recording():
        return

    if should_send_prompts():
        if kwargs.get("prompt") is not None:
            set_span_attribute(
                span, f"{SpanAttributes.LLM_PROMPTS}.0.user", kwargs.get("prompt")
            )

        elif kwargs.get("messages") is not None:
            for i, message in enumerate(kwargs.get("messages")):
                set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                    _dump_content(message.get("content")),
                )
                set_span_attribute(
                    span, f"{SpanAttributes.LLM_PROMPTS}.{i}.role", message.get("role")
                )


@dont_throw
def set_model_input_attributes(span, kwargs):
    if not span.is_recording():
        return

    set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
    set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_tokens_to_sample")
    )
    set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TEMPERATURE, kwargs.get("temperature")
    )
    set_span_attribute(span, SpanAttributes.LLM_REQUEST_TOP_P, kwargs.get("top_p"))
    set_span_attribute(
        span, SpanAttributes.LLM_FREQUENCY_PENALTY, kwargs.get("frequency_penalty")
    )
    set_span_attribute(
        span, SpanAttributes.LLM_PRESENCE_PENALTY, kwargs.get("presence_penalty")
    )
    set_span_attribute(
        span, SpanAttributes.LLM_IS_STREAMING, kwargs.get("stream") or False
    )


def set_streaming_response_attributes(
    span, accumulated_content, finish_reason=None, usage=None
):
    """Set span attributes for accumulated streaming response."""
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
            span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, usage.completion_tokens
        )
        set_span_attribute(
            span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.prompt_tokens
        )
        set_span_attribute(
            span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.total_tokens
        )


@dont_throw
def set_model_response_attributes(span, response, token_histogram):
    if not span.is_recording():
        return
    response = model_as_dict(response)
    set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, response.get("model"))
    set_span_attribute(span, GEN_AI_RESPONSE_ID, response.get("id"))

    usage = response.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if usage:
        set_span_attribute(
            span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.get("total_tokens")
        )
        set_span_attribute(
            span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens
        )
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
                SpanAttributes.LLM_RESPONSE_MODEL: response.get("model"),
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
                SpanAttributes.LLM_RESPONSE_MODEL: response.get("model"),
            },
        )


def set_response_attributes(span, response):
    if not span.is_recording():
        return
    choices = model_as_dict(response).get("choices")
    if should_send_prompts() and choices:
        _set_completions(span, choices)


def _set_completions(span, choices):
    if choices is None or not should_send_prompts():
        return

    for choice in choices:
        index = choice.get("index")
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
        set_span_attribute(span, f"{prefix}.finish_reason", choice.get("finish_reason"))

        if choice.get("content_filter_results"):
            set_span_attribute(
                span,
                f"{prefix}.{CONTENT_FILTER_KEY}",
                json.dumps(choice.get("content_filter_results")),
            )

        if choice.get("finish_reason") == "content_filter":
            set_span_attribute(span, f"{prefix}.role", "assistant")
            set_span_attribute(span, f"{prefix}.content", "FILTERED")

            return

        message = choice.get("message")
        if not message:
            return

        set_span_attribute(span, f"{prefix}.role", message.get("role"))
        set_span_attribute(span, f"{prefix}.content", message.get("content"))

        function_call = message.get("function_call")
        if function_call:
            set_span_attribute(
                span, f"{prefix}.tool_calls.0.name", function_call.get("name")
            )
            set_span_attribute(
                span,
                f"{prefix}.tool_calls.0.arguments",
                function_call.get("arguments"),
            )

        tool_calls = message.get("tool_calls")
        if tool_calls:
            for i, tool_call in enumerate(tool_calls):
                function = tool_call.get("function")
                set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.id",
                    tool_call.get("id"),
                )
                set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.name",
                    function.get("name"),
                )
                set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.arguments",
                    function.get("arguments"),
                )


def _dump_content(content):
    if isinstance(content, str):
        return content
    json_serializable = []
    for item in content:
        if item.get("type") == "text":
            json_serializable.append({"type": "text", "text": item.get("text")})
        elif item.get("type") == "image":
            json_serializable.append(
                {
                    "type": "image",
                    "source": {
                        "type": item.get("source").get("type"),
                        "media_type": item.get("source").get("media_type"),
                        "data": str(item.get("source").get("data")),
                    },
                }
            )
    return json.dumps(json_serializable)
