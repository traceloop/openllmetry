import json
from collections.abc import Mapping

from opentelemetry.instrumentation.ollama.utils import dont_throw, should_send_prompts
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
def set_input_attributes(span, llm_request_type, kwargs):
    if not span.is_recording():
        return
    if should_send_prompts():
        json_data = kwargs.get("json", {})

        if llm_request_type == LLMRequestTypeValues.CHAT:
            _set_span_attribute(span, f"{GenAIAttributes.GEN_AI_PROMPT}.0.role", "user")
            for index, message in enumerate(json_data.get("messages")):
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
            _set_prompts(span, json_data.get("messages"))
            if json_data.get("tools"):
                set_tools_attributes(span, json_data.get("tools"))
        else:
            _set_span_attribute(span, f"{GenAIAttributes.GEN_AI_PROMPT}.0.role", "user")
            _set_span_attribute(
                span, f"{GenAIAttributes.GEN_AI_PROMPT}.0.content", json_data.get("prompt")
            )


_REQUEST_OPTION_ATTRIBUTES = {
    "temperature": GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE,
    "top_p": GenAIAttributes.GEN_AI_REQUEST_TOP_P,
    "top_k": GenAIAttributes.GEN_AI_REQUEST_TOP_K,
    "num_predict": GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS,
    "seed": GenAIAttributes.GEN_AI_REQUEST_SEED,
    "frequency_penalty": GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY,
    "presence_penalty": GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY,
}


def _set_request_options(span, options):
    if not isinstance(options, Mapping):
        return
    for option, attribute in _REQUEST_OPTION_ATTRIBUTES.items():
        value = options.get(option)
        if option == "num_predict" and isinstance(value, (int, float)) and value < 0:
            continue
        _set_span_attribute(span, attribute, value)

    stop = options.get("stop")
    if isinstance(stop, str):
        stop = (stop,)
    if isinstance(stop, (list, tuple)):
        stop = tuple(s for s in stop if isinstance(s, str) and s)
        if stop:
            span.set_attribute(GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES, stop)


@dont_throw
def set_model_input_attributes(span, kwargs):
    if not span.is_recording():
        return
    json_data = kwargs.get("json", {})
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MODEL, json_data.get("model"))
    _set_request_options(span, json_data.get("options"))
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
                f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content",
                response.get("response"),
            )
            _set_span_attribute(
                span, f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role", "assistant"
            )
        elif llm_request_type == LLMRequestTypeValues.CHAT:
            index = 0
            prefix = f"{GenAIAttributes.GEN_AI_COMPLETION}.{index}"
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

    _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, response.get("model"))

    input_tokens = response.get("prompt_eval_count") or 0
    output_tokens = response.get("eval_count") or 0

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
    _set_span_attribute(span, GenAIAttributes.GEN_AI_SYSTEM, "Ollama")

    if (
        token_histogram is not None
        and isinstance(input_tokens, int)
        and input_tokens >= 0
    ):
        token_histogram.record(
            input_tokens,
            attributes={
                GenAIAttributes.GEN_AI_SYSTEM: "Ollama",
                GenAIAttributes.GEN_AI_TOKEN_TYPE: "input",
                GenAIAttributes.GEN_AI_RESPONSE_MODEL: response.get("model"),
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
                GenAIAttributes.GEN_AI_SYSTEM: "Ollama",
                GenAIAttributes.GEN_AI_TOKEN_TYPE: "output",
                GenAIAttributes.GEN_AI_RESPONSE_MODEL: response.get("model"),
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
        prefix = f"{GenAIAttributes.GEN_AI_PROMPT}.{i}"

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
