import json

from opentelemetry.instrumentation.ollama.utils import dont_throw, should_send_prompts
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
def set_input_attributes(span, llm_request_type, kwargs):
    if not span.is_recording():
        return
    if should_send_prompts():
        json_data = kwargs.get("json", {})

        if llm_request_type == LLMRequestTypeValues.CHAT:
            _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
            for index, message in enumerate(json_data.get("messages")):
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{index}.content",
                    message.get("content"),
                )
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{index}.role",
                    message.get("role"),
                )
            _set_prompts(span, json_data.get("messages"))
            if json_data.get("tools"):
                set_tools_attributes(span, json_data.get("tools"))
        else:
            _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_PROMPTS}.0.content", json_data.get("prompt")
            )


@dont_throw
def set_model_input_attributes(span, kwargs):
    if not span.is_recording():
        return
    json_data = kwargs.get("json", {})
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, json_data.get("model"))
    _set_span_attribute(
        span, SpanAttributes.LLM_IS_STREAMING, kwargs.get("stream") or False
    )


@dont_throw
def set_response_attributes(span, token_histogram, llm_request_type, response):
    if not span.is_recording():
        return

    if should_send_prompts():
        if llm_request_type == LLMRequestTypeValues.COMPLETION:
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
                response.get("response"),
            )
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant"
            )
        elif llm_request_type == LLMRequestTypeValues.CHAT:
            index = 0
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
            _set_span_attribute(
                span, f"{prefix}.content", response.get("message").get("content")
            )
            _set_span_attribute(
                span, f"{prefix}.role", response.get("message").get("role")
            )


@dont_throw
def set_model_response_attributes(span, token_histogram, llm_request_type, response):
    if llm_request_type == LLMRequestTypeValues.EMBEDDING or not span.is_recording():
        return

    _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, response.get("model"))

    input_tokens = response.get("prompt_eval_count") or 0
    output_tokens = response.get("eval_count") or 0

    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
        input_tokens + output_tokens,
    )
    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
        output_tokens,
    )
    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
        input_tokens,
    )
    _set_span_attribute(span, SpanAttributes.LLM_SYSTEM, "Ollama")

    if (
        token_histogram is not None
        and isinstance(input_tokens, int)
        and input_tokens >= 0
    ):
        token_histogram.record(
            input_tokens,
            attributes={
                SpanAttributes.LLM_SYSTEM: "Ollama",
                SpanAttributes.LLM_TOKEN_TYPE: "input",
                SpanAttributes.LLM_RESPONSE_MODEL: response.get("model"),
            },
        )

    if (
        token_histogram is not None
        and isinstance(output_tokens, int)
        and output_tokens >= 0
    ):
        token_histogram.record(
            output_tokens,
            attributes={
                SpanAttributes.LLM_SYSTEM: "Ollama",
                SpanAttributes.LLM_TOKEN_TYPE: "output",
                SpanAttributes.LLM_RESPONSE_MODEL: response.get("model"),
            },
        )


def set_tools_attributes(span, tools):
    if not tools:
        return

    for i, tool in enumerate(tools):
        function = tool.get("function")
        if not function:
            continue

        prefix = f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}"
        _set_span_attribute(span, f"{prefix}.name", function.get("name"))
        _set_span_attribute(span, f"{prefix}.description", function.get("description"))
        _set_span_attribute(
            span, f"{prefix}.parameters", json.dumps(function.get("parameters"))
        )


def _set_prompts(span, messages):
    if not span.is_recording() or messages is None:
        return
    if not should_send_prompts():
        return
    for i, msg in enumerate(messages):
        prefix = f"{SpanAttributes.LLM_PROMPTS}.{i}"

        _set_span_attribute(span, f"{prefix}.role", msg.get("role"))
        if msg.get("content"):
            content = msg.get("content")
            if isinstance(content, list):
                content = json.dumps(content)
            _set_span_attribute(span, f"{prefix}.content", content)
        if msg.get("tool_call_id"):
            _set_span_attribute(span, f"{prefix}.tool_call_id", msg.get("tool_call_id"))
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for i, tool_call in enumerate(tool_calls):
                function = tool_call.get("function")
                _set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.id",
                    tool_call.get("id"),
                )
                _set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.name",
                    function.get("name"),
                )
                # record arguments: ensure it's a JSON string for span attributes
                raw_args = function.get("arguments")
                if isinstance(raw_args, dict):
                    arg_str = json.dumps(raw_args)
                else:
                    arg_str = raw_args
                _set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.arguments",
                    arg_str,
                )
