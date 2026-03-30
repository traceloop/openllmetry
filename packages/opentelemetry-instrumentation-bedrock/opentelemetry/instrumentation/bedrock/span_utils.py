import json
import logging
import time

from opentelemetry.instrumentation.bedrock.config import Config
from opentelemetry.instrumentation.bedrock.utils import should_send_prompts
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes.aws_attributes import (
    AWS_BEDROCK_GUARDRAIL_ID
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiOperationNameValues,
    GenAiSystemValues,
)
from opentelemetry.semconv_ai import SpanAttributes

logger = logging.getLogger(__name__)

PROMPT_FILTER_KEY = "prompt_filter_results"
CONTENT_FILTER_KEY = "content_filter_results"

# Bedrock guardrail attribute keys (vendor namespace, replacing deprecated gen_ai.prompt/gen_ai.completion prefix)
BEDROCK_GUARDRAIL_INPUT_FILTER = "gen_ai.bedrock.guardrail.input_filter"
BEDROCK_GUARDRAIL_OUTPUT_FILTER = "gen_ai.bedrock.guardrail.output_filter"

# Bedrock finish reason → OTel GenAI enum mapping
# OTel values: stop, length, content_filter, tool_call, error
BEDROCK_FINISH_REASON_MAP = {
    # Anthropic via Bedrock
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_call",
    "max_tokens": "length",
    # Cohere via Bedrock
    "COMPLETE": "stop",
    "TOOL_CALL": "tool_call",
    "MAX_TOKENS": "length",
    "ERROR": "error",
    # Amazon Titan
    "FINISH": "stop",
    # AI21
    "endoftext": "stop",
    # Converse API
    "guardrail_intervened": "content_filter",
}


def _map_finish_reason(reason):
    """Map provider-specific finish reason to OTel GenAI enum value."""
    if not reason:
        return None
    mapped = BEDROCK_FINISH_REASON_MAP.get(reason)
    if mapped is None:
        logger.warning(
            "Unmapped Bedrock finish reason '%s' — passing through verbatim. "
            "Consider adding it to BEDROCK_FINISH_REASON_MAP.",
            reason,
        )
        return reason
    return mapped


def _text_part(content):
    """Create a text part for the parts array."""
    return {"type": "text", "content": content}


def _output_message(role, parts, finish_reason=None):
    """Create an output message dict. finish_reason is required per OTel spec."""
    return {"role": role, "parts": parts, "finish_reason": finish_reason if finish_reason else ""}


def _anthropic_content_to_parts(content_blocks):
    """Convert Anthropic content blocks to OTel parts format."""
    parts = []
    for block in content_blocks:
        if isinstance(block, str):
            parts.append(_text_part(block))
        elif isinstance(block, dict):
            block_type = block.get("type", "text")
            if block_type == "text":
                parts.append(_text_part(block.get("text", "")))
            elif block_type == "tool_use":
                raw_input = block.get("input", {})
                if isinstance(raw_input, str):
                    try:
                        raw_input = json.loads(raw_input)
                    except (json.JSONDecodeError, TypeError):
                        pass
                parts.append({
                    "type": "tool_call",
                    "name": block.get("name"),
                    "id": block.get("id"),
                    "arguments": raw_input,
                })
            elif block_type == "tool_result":
                parts.append({
                    "type": "tool_call_response",
                    "id": block.get("tool_use_id"),
                    "response": block.get("content", ""),
                })
            elif block_type == "image":
                source = block.get("source", {})
                source_type = source.get("type")
                if source_type == "base64":
                    parts.append({
                        "type": "blob",
                        "modality": "image",
                        "mime_type": source.get("media_type", ""),
                        "content": source.get("data", ""),
                    })
                elif source_type == "url":
                    parts.append({
                        "type": "uri",
                        "modality": "image",
                        "uri": source.get("url", ""),
                    })
                else:
                    parts.append({
                        "type": "blob",
                        "modality": "image",
                        "mime_type": source.get("media_type", ""),
                        "content": source.get("data", ""),
                    })
            elif block_type == "thinking":
                parts.append({"type": "reasoning", "content": block.get("thinking", "")})
            else:
                # Stopgap: unknown Anthropic block types are encoded as text
                # to avoid emitting non-OTel part types.
                parts.append({"type": "text", "content": json.dumps(block)})
        else:
            parts.append(_text_part(str(block)))
    return parts


_anthropic_client = [None]  # mutable container for lazy-initialized client


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def set_model_message_span_attributes(model_vendor, span, request_body):
    if not should_send_prompts():
        return
    if model_vendor == "cohere":
        _set_prompt_span_attributes(span, request_body)
    elif model_vendor == "anthropic":
        if "prompt" in request_body:
            _set_prompt_span_attributes(span, request_body)
        elif "messages" in request_body:
            if "system" in request_body:
                system_val = request_body["system"]
                if isinstance(system_val, str):
                    system_parts = [_text_part(system_val)]
                elif isinstance(system_val, list):
                    system_parts = _anthropic_content_to_parts(system_val)
                else:
                    system_parts = [_text_part(json.dumps(system_val))]
                _set_span_attribute(
                    span,
                    GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS,
                    json.dumps(system_parts),
                )
            input_messages = []
            for message in request_body.get("messages"):
                content = message.get("content")
                if isinstance(content, str):
                    parts = [_text_part(content)]
                elif isinstance(content, list):
                    parts = _anthropic_content_to_parts(content)
                else:
                    parts = [_text_part(json.dumps(content))]
                input_messages.append({
                    "role": message.get("role"),
                    "parts": parts,
                })
            _set_span_attribute(
                span,
                GenAIAttributes.GEN_AI_INPUT_MESSAGES,
                json.dumps(input_messages),
            )
    elif model_vendor == "ai21":
        _set_prompt_span_attributes(span, request_body)
    elif model_vendor == "meta":
        _set_llama_prompt_span_attributes(span, request_body)
    elif model_vendor == "amazon":
        _set_amazon_input_span_attributes(span, request_body)
    elif model_vendor == "imported_model":
        _set_imported_model_prompt_span_attributes(span, request_body)


def set_model_choice_span_attributes(model_vendor, span, response_body):
    _set_finish_reasons_unconditionally(model_vendor, span, response_body)
    if not should_send_prompts():
        return
    if model_vendor == "cohere":
        _set_generations_span_attributes(span, response_body)
    elif model_vendor == "anthropic":
        _set_anthropic_response_span_attributes(span, response_body)
    elif model_vendor == "ai21":
        _set_span_completions_attributes(span, response_body)
    elif model_vendor == "meta":
        _set_llama_response_span_attributes(span, response_body)
    elif model_vendor == "amazon":
        _set_amazon_response_span_attributes(span, response_body)
    elif model_vendor == "imported_model":
        _set_imported_model_response_span_attributes(span, response_body)


def _set_finish_reasons_unconditionally(model_vendor, span, response_body):
    """Set finish_reasons on span regardless of should_send_prompts() — it's metadata, not content."""
    finish_reasons = []
    if model_vendor == "cohere":
        for gen in response_body.get("generations", []):
            fr = _map_finish_reason(gen.get("finish_reason"))
            if fr:
                finish_reasons.append(fr)
    elif model_vendor == "anthropic":
        fr = _map_finish_reason(response_body.get("stop_reason"))
        if fr:
            finish_reasons.append(fr)
    elif model_vendor == "ai21":
        for comp in response_body.get("completions", []):
            fr_data = comp.get("finishReason", {})
            raw = fr_data.get("reason") if isinstance(fr_data, dict) else str(fr_data)
            fr = _map_finish_reason(raw)
            if fr:
                finish_reasons.append(fr)
    elif model_vendor == "meta":
        fr = _map_finish_reason(response_body.get("stop_reason"))
        if fr:
            finish_reasons.append(fr)
    elif model_vendor == "amazon":
        if "results" in response_body:
            for result in response_body.get("results", []):
                fr = _map_finish_reason(result.get("completionReason"))
                if fr:
                    finish_reasons.append(fr)
        elif "output" in response_body:
            fr = _map_finish_reason(response_body.get("stopReason"))
            if fr:
                finish_reasons.append(fr)
        else:
            fr = _map_finish_reason(response_body.get("completionReason"))
            if fr:
                finish_reasons.append(fr)
    elif model_vendor == "imported_model":
        fr = _map_finish_reason(response_body.get("stop_reason"))
        if fr:
            finish_reasons.append(fr)
    if finish_reasons:
        _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, tuple(finish_reasons))


def set_model_span_attributes(
    provider,
    model_vendor,
    model,
    span,
    request_body,
    response_body,
    headers,
    metric_params,
    kwargs,
):
    response_model = response_body.get("model")
    response_id = response_body.get("id")

    _set_span_attribute(span, AWS_BEDROCK_GUARDRAIL_ID, _guardrail_value(kwargs))

    _set_span_attribute(span, GenAIAttributes.GEN_AI_PROVIDER_NAME, provider)
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MODEL, model)
    _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, response_model)
    _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, response_id)

    if model_vendor == "cohere":
        _set_cohere_span_attributes(span, request_body, response_body, metric_params)
    elif model_vendor == "anthropic":
        if "prompt" in request_body:
            _set_anthropic_completion_span_attributes(
                span, request_body, response_body, headers, metric_params
            )
        elif "messages" in request_body:
            _set_anthropic_messages_span_attributes(
                span, request_body, response_body, headers, metric_params
            )
    elif model_vendor == "ai21":
        _set_ai21_span_attributes(span, request_body, response_body, metric_params)
    elif model_vendor == "meta":
        _set_llama_span_attributes(span, request_body, response_body, metric_params)
    elif model_vendor == "amazon":
        _set_amazon_span_attributes(
            span, request_body, response_body, headers, metric_params
        )
    elif model_vendor == "imported_model":
        _set_imported_model_span_attributes(
            span, request_body, response_body, metric_params
        )


def _guardrail_value(request_body):
    identifier = request_body.get("guardrailIdentifier")
    if identifier is not None:
        version = request_body.get("guardrailVersion")
        return f"{identifier}:{version}"
    return None


def set_guardrail_attributes(span, input_filters, output_filters):
    if input_filters:
        _set_span_attribute(
            span,
            BEDROCK_GUARDRAIL_INPUT_FILTER,
            json.dumps(input_filters, default=str)
        )
    if output_filters:
        _set_span_attribute(
            span,
            BEDROCK_GUARDRAIL_OUTPUT_FILTER,
            json.dumps(output_filters, default=str)
        )


def _set_prompt_span_attributes(span, request_body):
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_INPUT_MESSAGES,
        json.dumps([{"role": "user", "parts": [_text_part(request_body.get("prompt"))]}]),
    )


def _set_cohere_span_attributes(span, request_body, response_body, metric_params):
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_P, request_body.get("p"))
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, request_body.get("temperature")
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS, request_body.get("max_tokens")
    )

    # based on contract at
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command-r-plus.html
    input_tokens = response_body.get("token_count", {}).get("prompt_tokens")
    output_tokens = response_body.get("token_count", {}).get("response_tokens")

    if input_tokens is None or output_tokens is None:
        meta = response_body.get("meta", {})
        billed_units = meta.get("billed_units", {})
        input_tokens = input_tokens or billed_units.get("input_tokens")
        output_tokens = output_tokens or billed_units.get("output_tokens")

    if input_tokens is not None and output_tokens is not None:
        _record_usage_to_span(
            span,
            input_tokens,
            output_tokens,
            metric_params,
        )


def _set_generations_span_attributes(span, response_body):
    output_messages = []
    finish_reasons = []
    for generation in response_body.get("generations"):
        fr = _map_finish_reason(generation.get("finish_reason"))
        finish_reasons.append(fr)
        output_messages.append(_output_message("assistant", [_text_part(generation.get("text"))], fr))
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
        json.dumps(output_messages),
    )


def _set_anthropic_completion_span_attributes(
    span, request_body, response_body, headers, metric_params
):
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_TOP_P, request_body.get("top_p")
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, request_body.get("temperature")
    )
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS,
        request_body.get("max_tokens_to_sample"),
    )

    if (
        response_body.get("usage") is not None
        and response_body.get("usage").get("input_tokens") is not None
        and response_body.get("usage").get("output_tokens") is not None
    ):
        _record_usage_to_span(
            span,
            response_body.get("usage").get("input_tokens"),
            response_body.get("usage").get("output_tokens"),
            metric_params,
        )
    elif response_body.get("invocation_metrics") is not None:
        _record_usage_to_span(
            span,
            response_body.get("invocation_metrics").get("inputTokenCount"),
            response_body.get("invocation_metrics").get("outputTokenCount"),
            metric_params,
        )
    elif headers and headers.get("x-amzn-bedrock-input-token-count") is not None:
        # For Anthropic V2 models (claude-v2), token counts are in HTTP headers
        input_tokens = int(headers.get("x-amzn-bedrock-input-token-count", 0))
        output_tokens = int(headers.get("x-amzn-bedrock-output-token-count", 0))
        _record_usage_to_span(
            span,
            input_tokens,
            output_tokens,
            metric_params,
        )
    elif Config.enrich_token_usage:
        _record_usage_to_span(
            span,
            _count_anthropic_tokens([request_body.get("prompt")]),
            _count_anthropic_tokens([response_body.get("completion")]),
            metric_params,
        )


def _set_anthropic_response_span_attributes(span, response_body):
    fr = _map_finish_reason(response_body.get("stop_reason"))
    if response_body.get("completion") is not None:
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps([_output_message("assistant", [_text_part(response_body.get("completion"))], fr)]),
        )
    elif response_body.get("content") is not None:
        parts = _anthropic_content_to_parts(response_body.get("content"))
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps([_output_message("assistant", parts, fr)]),
        )


def _set_anthropic_messages_span_attributes(
    span, request_body, response_body, headers, metric_params
):
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_TOP_P, request_body.get("top_p")
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, request_body.get("temperature")
    )
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS,
        request_body.get("max_tokens"),
    )

    prompt_tokens = 0
    completion_tokens = 0
    if (
        response_body.get("usage") is not None
        and response_body.get("usage").get("input_tokens") is not None
        and response_body.get("usage").get("output_tokens") is not None
    ):
        prompt_tokens = response_body.get("usage").get("input_tokens")
        completion_tokens = response_body.get("usage").get("output_tokens")
        _record_usage_to_span(span, prompt_tokens, completion_tokens, metric_params)
    elif response_body.get("invocation_metrics") is not None:
        prompt_tokens = response_body.get("invocation_metrics").get("inputTokenCount")
        completion_tokens = response_body.get("invocation_metrics").get(
            "outputTokenCount"
        )
        _record_usage_to_span(span, prompt_tokens, completion_tokens, metric_params)
    elif headers and headers.get("x-amzn-bedrock-input-token-count") is not None:
        # For Anthropic V2 models (claude-v2), token counts are in HTTP headers
        prompt_tokens = int(headers.get("x-amzn-bedrock-input-token-count", 0))
        completion_tokens = int(headers.get("x-amzn-bedrock-output-token-count", 0))
        _record_usage_to_span(span, prompt_tokens, completion_tokens, metric_params)
    elif Config.enrich_token_usage:
        messages = [message.get("content") for message in request_body.get("messages")]

        raw_messages = []
        for message in messages:
            if isinstance(message, str):
                raw_messages.append(message)
            else:
                raw_messages.extend([content.get("text") for content in message])
        prompt_tokens = _count_anthropic_tokens(raw_messages)
        completion_tokens = _count_anthropic_tokens(
            [content.get("text") for content in response_body.get("content")]
        )
        _record_usage_to_span(span, prompt_tokens, completion_tokens, metric_params)


def _count_anthropic_tokens(messages: list[str]):
    try:
        import anthropic
    except ImportError:
        logger.debug("anthropic package not installed — skipping token counting")
        return 0

    # Lazy initialization of the Anthropic client
    if _anthropic_client[0] is None:
        try:
            _anthropic_client[0] = anthropic.Anthropic()
        except Exception as e:
            logger.debug(f"Failed to initialize Anthropic client for token counting: {e}")
            return 0

    count = 0
    try:
        for message in messages:
            count += _anthropic_client[0].count_tokens(text=message)
    except Exception as e:
        logger.debug(f"Failed to count tokens with Anthropic client: {e}")
        return 0

    return count


def _set_ai21_span_attributes(span, request_body, response_body, metric_params):
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_TOP_P, request_body.get("topP")
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, request_body.get("temperature")
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS, request_body.get("maxTokens")
    )

    _record_usage_to_span(
        span,
        len(response_body.get("prompt").get("tokens")),
        len(response_body.get("completions")[0].get("data").get("tokens")),
        metric_params,
    )


def _set_span_completions_attributes(span, response_body):
    output_messages = []
    finish_reasons = []
    for completion in response_body.get("completions"):
        fr_data = completion.get("finishReason", {})
        raw_reason = fr_data.get("reason") if isinstance(fr_data, dict) else str(fr_data) or None
        fr = _map_finish_reason(raw_reason)
        if fr:
            finish_reasons.append(fr)
        output_messages.append(_output_message("assistant", [_text_part(completion.get("data").get("text"))], fr))
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
        json.dumps(output_messages),
    )


def _set_llama_span_attributes(span, request_body, response_body, metric_params):
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_TOP_P, request_body.get("top_p")
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, request_body.get("temperature")
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS, request_body.get("max_gen_len")
    )

    _record_usage_to_span(
        span,
        response_body.get("prompt_token_count"),
        response_body.get("generation_token_count"),
        metric_params,
    )


def _set_llama_prompt_span_attributes(span, request_body):
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_INPUT_MESSAGES,
        json.dumps([{"role": "user", "parts": [_text_part(request_body.get("prompt"))]}]),
    )


def _set_llama_response_span_attributes(span, response_body):
    fr = _map_finish_reason(response_body.get("stop_reason"))
    if response_body.get("generation"):
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps([_output_message("assistant", [_text_part(response_body.get("generation"))], fr)]),
        )
    else:
        output_messages = []
        for generation in response_body.get("generations"):
            output_messages.append(_output_message("assistant", [_text_part(generation)], fr))
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps(output_messages),
        )


def _set_amazon_span_attributes(
    span, request_body, response_body, headers, metric_params
):
    if "textGenerationConfig" in request_body:
        config = request_body.get("textGenerationConfig", {})
        _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_P, config.get("topP"))
        _set_span_attribute(
            span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, config.get("temperature")
        )
        _set_span_attribute(
            span, GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS, config.get("maxTokenCount")
        )
    elif "inferenceConfig" in request_body:
        config = request_body.get("inferenceConfig", {})
        _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_P, config.get("topP"))
        _set_span_attribute(
            span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, config.get("temperature")
        )
        _set_span_attribute(
            span, GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS, config.get("maxTokens")
        )

    total_completion_tokens = 0
    total_prompt_tokens = 0
    if "results" in response_body:
        total_prompt_tokens = int(response_body.get("inputTextTokenCount", 0))
        for result in response_body.get("results"):
            if "tokenCount" in result:
                total_completion_tokens += int(result.get("tokenCount", 0))
            elif "totalOutputTextTokenCount" in result:
                total_completion_tokens += int(
                    result.get("totalOutputTextTokenCount", 0)
                )
    elif "usage" in response_body:
        total_prompt_tokens += int(response_body.get("inputTokens", 0))
        total_completion_tokens += int(
            headers.get("x-amzn-bedrock-output-token-count", 0)
        )
    # checks for Titan models
    if "inputTextTokenCount" in response_body:
        total_prompt_tokens = response_body.get("inputTextTokenCount")
    if "totalOutputTextTokenCount" in response_body:
        total_completion_tokens = response_body.get("totalOutputTextTokenCount")

    _record_usage_to_span(
        span,
        total_prompt_tokens,
        total_completion_tokens,
        metric_params,
    )


def _set_amazon_input_span_attributes(span, request_body):
    if "inputText" in request_body:
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_INPUT_MESSAGES,
            json.dumps([{"role": "user", "parts": [_text_part(request_body.get("inputText"))]}]),
        )
    else:
        if "system" in request_body:
            _set_span_attribute(
                span,
                GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS,
                json.dumps([_text_part(prompt.get("text")) for prompt in request_body["system"]]),
            )
        input_messages = []
        for prompt in request_body["messages"]:
            content = prompt.get("content", "")
            if isinstance(content, str):
                parts = [_text_part(content)]
            elif isinstance(content, list):
                parts = _converse_content_to_parts(content)
            else:
                parts = [_text_part(json.dumps(content, default=str))]
            input_messages.append({
                "role": prompt.get("role"),
                "parts": parts,
            })
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_INPUT_MESSAGES,
            json.dumps(input_messages),
        )


def _set_amazon_response_span_attributes(span, response_body):
    if "results" in response_body:
        output_messages = []
        for result in response_body.get("results"):
            fr = _map_finish_reason(result.get("completionReason"))
            output_messages.append(_output_message("assistant", [_text_part(result.get("outputText"))], fr))
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps(output_messages),
        )
    elif "outputText" in response_body:
        fr = _map_finish_reason(response_body.get("completionReason"))
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps([_output_message("assistant", [_text_part(response_body.get("outputText"))], fr)]),
        )
    elif "output" in response_body:
        fr = _map_finish_reason(response_body.get("stopReason"))
        content_blocks = response_body.get("output", {}).get("message", {}).get("content", [])
        parts = _converse_content_to_parts(content_blocks)
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps([_output_message("assistant", parts, fr)]),
        )


def _set_imported_model_span_attributes(
    span, request_body, response_body, metric_params
):
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_TOP_P, request_body.get("topP")
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, request_body.get("temperature")
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS, request_body.get("max_tokens")
    )
    prompt_tokens = (
        response_body.get("usage", {}).get("prompt_tokens")
        if response_body.get("usage", {}).get("prompt_tokens") is not None
        else response_body.get("prompt_token_count")
    )
    completion_tokens = response_body.get("usage", {}).get(
        "completion_tokens"
    ) or response_body.get("generation_token_count")

    _record_usage_to_span(
        span,
        prompt_tokens,
        completion_tokens,
        metric_params,
    )


def _set_imported_model_response_span_attributes(span, response_body):
    fr = _map_finish_reason(response_body.get("stop_reason"))
    content = response_body.get("generation")
    if content is None and response_body.get("choices"):
        content = response_body["choices"][0].get("text")
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
        json.dumps([_output_message("assistant", [_text_part(content)], fr)]),
    )


def _set_imported_model_prompt_span_attributes(span, request_body):
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_INPUT_MESSAGES,
        json.dumps([{"role": "user", "parts": [_text_part(request_body.get("prompt"))]}]),
    )


def _record_usage_to_span(span, prompt_tokens, completion_tokens, metric_params):
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
        prompt_tokens,
    )
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
        completion_tokens,
    )
    _set_span_attribute(
        span,
        SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS,
        prompt_tokens + completion_tokens,
    )

    metric_attributes = _metric_shared_attributes(
        metric_params.vendor, metric_params.model, metric_params.is_stream
    )

    if metric_params.duration_histogram:
        duration = time.time() - metric_params.start_time
        metric_params.duration_histogram.record(
            duration,
            attributes=metric_attributes,
        )

    if (
        metric_params.token_histogram
        and type(prompt_tokens) is int
        and prompt_tokens >= 0
    ):
        metric_params.token_histogram.record(
            prompt_tokens,
            attributes={
                **metric_attributes,
                GenAIAttributes.GEN_AI_TOKEN_TYPE: "input",
            },
        )
    if (
        metric_params.token_histogram
        and type(completion_tokens) is int
        and completion_tokens >= 0
    ):
        metric_params.token_histogram.record(
            completion_tokens,
            attributes={
                **metric_attributes,
                GenAIAttributes.GEN_AI_TOKEN_TYPE: "output",
            },
        )


def _metric_shared_attributes(
    response_vendor: str, response_model: str, is_streaming: bool = False
):
    return {
        GenAIAttributes.GEN_AI_RESPONSE_MODEL: response_model,
        GenAIAttributes.GEN_AI_PROVIDER_NAME: GenAiSystemValues.AWS_BEDROCK.value,
    }


def set_converse_model_span_attributes(span, provider, model, kwargs):
    _set_span_attribute(span, GenAIAttributes.GEN_AI_PROVIDER_NAME, provider)
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MODEL, model)
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_OPERATION_NAME, GenAiOperationNameValues.CHAT.value
    )

    guardrail_config = kwargs.get("guardrailConfig")
    if guardrail_config:
        _set_span_attribute(span, AWS_BEDROCK_GUARDRAIL_ID, _guardrail_value(guardrail_config))

    config = {}
    if "inferenceConfig" in kwargs:
        config = kwargs.get("inferenceConfig")

    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_P, config.get("topP"))
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, config.get("temperature")
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS, config.get("maxTokens")
    )

    tool_config = kwargs.get("toolConfig", {})
    tools = tool_config.get("tools", []) if tool_config else []
    if tools:
        tool_defs = []
        for tool in tools:
            spec = tool.get("toolSpec", {}) if isinstance(tool, dict) else {}
            if not spec:
                continue
            tool_def = {"name": spec.get("name", "")}
            if "description" in spec:
                tool_def["description"] = spec.get("description")
            if "inputSchema" in spec:
                tool_def["parameters"] = spec.get("inputSchema", {}).get("json", {})
            tool_defs.append(tool_def)
        if tool_defs:
            _set_span_attribute(
                span,
                GenAIAttributes.GEN_AI_TOOL_DEFINITIONS,
                json.dumps(tool_defs),
            )


def _converse_content_to_parts(content_blocks):
    """Convert Bedrock Converse API content blocks to OTel parts format."""
    parts = []
    for block in content_blocks:
        if isinstance(block, str):
            parts.append(_text_part(block))
        elif isinstance(block, dict):
            if "text" in block:
                parts.append(_text_part(block["text"]))
            elif "toolUse" in block:
                tool = block["toolUse"]
                parts.append({
                    "type": "tool_call",
                    "name": tool.get("name"),
                    "id": tool.get("toolUseId"),
                    "arguments": tool.get("input", {}),
                })
            elif "toolResult" in block:
                result = block["toolResult"]
                content = result.get("content", [])
                if isinstance(content, list):
                    text_parts = [item["text"] for item in content if isinstance(item, dict) and "text" in item]
                    response_str = " ".join(text_parts) if text_parts else json.dumps(content, default=str)
                else:
                    response_str = str(content)
                parts.append({
                    "type": "tool_call_response",
                    "id": result.get("toolUseId"),
                    "response": response_str,
                })
            elif "image" in block:
                img = block["image"]
                fmt = img.get("format", "")
                # BlobPart.content left empty to avoid bloating spans with
                # base64-encoded image data. Schema requires the field but
                # empty string satisfies the constraint.
                parts.append({
                    "type": "blob",
                    "modality": "image",
                    "mime_type": f"image/{fmt}" if fmt else "",
                    "content": "",
                })
            elif "video" in block:
                vid = block["video"]
                fmt = vid.get("format", "")
                # BlobPart.content left empty — same rationale as image blocks.
                parts.append({
                    "type": "blob",
                    "modality": "video",
                    "mime_type": f"video/{fmt}" if fmt else "",
                    "content": "",
                })
            elif "document" in block:
                doc = block["document"]
                # Converse API uses short format names (e.g. "pdf", "csv");
                # map to proper MIME types for OTel semconv compliance.
                fmt = doc.get("format", "")
                mime_map = {
                    "pdf": "application/pdf",
                    "csv": "text/csv",
                    "doc": "application/msword",
                    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "xls": "application/vnd.ms-excel",
                    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "html": "text/html",
                    "txt": "text/plain",
                    "md": "text/markdown",
                }
                mime_type = mime_map.get(fmt, f"application/{fmt}" if fmt else "")
                # BlobPart.content is for base64-encoded binary data; left empty to
                # avoid bloating spans. Document name stored as additional property
                # (additionalProperties: true in BlobPart schema).
                part = {
                    "type": "blob",
                    "modality": "document",
                    "mime_type": mime_type,
                    "content": "",
                }
                name = doc.get("name")
                if name:
                    part["name"] = name
                parts.append(part)
            elif "guardContent" in block:
                parts.append({"type": "text", "content": json.dumps(block, default=str)})
            else:
                parts.append({"type": "text", "content": json.dumps(block, default=str)})
        else:
            parts.append(_text_part(str(block)))
    return parts


def set_converse_input_prompt_span_attributes(kwargs, span):
    if not should_send_prompts():
        return
    if "system" in kwargs:
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS,
            json.dumps([_text_part(prompt.get("text")) for prompt in kwargs["system"]]),
        )
    input_messages = []
    if "messages" in kwargs:
        for prompt in kwargs["messages"]:
            content = prompt.get("content", "")
            if isinstance(content, str):
                parts = [_text_part(content)]
            elif isinstance(content, list):
                parts = _converse_content_to_parts(content)
            else:
                parts = [_text_part(json.dumps(content, default=str))]
            input_messages.append({
                "role": prompt.get("role"),
                "parts": parts,
            })
    if input_messages:
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_INPUT_MESSAGES,
            json.dumps(input_messages),
        )


def _set_converse_finish_reasons(span, stop_reason):
    fr = _map_finish_reason(stop_reason)
    if fr:
        _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, (fr,))
    return fr


def set_converse_response_span_attributes(response, span):
    if "output" in response:
        fr = _set_converse_finish_reasons(span, response.get("stopReason"))
        if not should_send_prompts():
            return
        message = response["output"]["message"]
        parts = _converse_content_to_parts(message.get("content", []))
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps([_output_message(message.get("role"), parts, fr)]),
        )


def set_converse_streaming_response_span_attributes(
    response,
    role,
    span,
    finish_reason=None,
    tool_blocks=None,
    reasoning_blocks=None,
):
    fr = _set_converse_finish_reasons(span, finish_reason)
    if not should_send_prompts():
        return
    parts = []
    reasoning_content = "".join(reasoning_blocks or [])
    if reasoning_content:
        parts.append({"type": "reasoning", "content": reasoning_content})
    text_content = "".join(response)
    if text_content:
        parts.append(_text_part(text_content))
    for tool in tool_blocks or []:
        raw_input = tool.get("input", {})
        if isinstance(raw_input, str):
            try:
                raw_input = json.loads(raw_input)
            except (json.JSONDecodeError, TypeError):
                pass
        parts.append({
            "type": "tool_call",
            "name": tool.get("name"),
            "id": tool.get("toolUseId"),
            "arguments": raw_input,
        })
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
        json.dumps([_output_message(role, parts, fr)]),
    )


def converse_usage_record(span, response, metric_params):
    if "usage" not in response:
        return

    prompt_tokens = response["usage"].get("inputTokens", 0)
    completion_tokens = response["usage"].get("outputTokens", 0)

    _record_usage_to_span(
        span,
        prompt_tokens,
        completion_tokens,
        metric_params,
    )
