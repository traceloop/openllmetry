import json
import time

import anthropic
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

PROMPT_FILTER_KEY = "prompt_filter_results"
CONTENT_FILTER_KEY = "content_filter_results"

anthropic_client = None


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
            input_messages = []
            for message in request_body.get("messages"):
                input_messages.append({
                    "role": message.get("role"),
                    "content": json.dumps(message.get("content")),
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
            f"{GenAIAttributes.GEN_AI_PROMPT}.{PROMPT_FILTER_KEY}",
            json.dumps(input_filters, default=str)
        )
    if output_filters:
        _set_span_attribute(
            span,
            f"{GenAIAttributes.GEN_AI_COMPLETION}.{CONTENT_FILTER_KEY}",
            json.dumps(output_filters, default=str)
        )


def _set_prompt_span_attributes(span, request_body):
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_INPUT_MESSAGES,
        json.dumps([{"role": "user", "content": request_body.get("prompt")}]),
    )


def _set_cohere_span_attributes(span, request_body, response_body, metric_params):
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_OPERATION_NAME, GenAiOperationNameValues.TEXT_COMPLETION.value
    )
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
    for generation in response_body.get("generations"):
        output_messages.append({
            "role": "assistant",
            "content": generation.get("text"),
        })
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
        json.dumps(output_messages),
    )


def _set_anthropic_completion_span_attributes(
    span, request_body, response_body, headers, metric_params
):
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_OPERATION_NAME, GenAiOperationNameValues.TEXT_COMPLETION.value
    )
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
    if response_body.get("completion") is not None:
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps([{"role": "assistant", "content": response_body.get("completion")}]),
        )
    elif response_body.get("content") is not None:
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps([{"role": "assistant", "content": json.dumps(response_body.get("content"))}]),
        )


def _set_anthropic_messages_span_attributes(
    span, request_body, response_body, headers, metric_params
):
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_OPERATION_NAME, GenAiOperationNameValues.CHAT.value
    )
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
    global anthropic_client

    # Lazy initialization of the Anthropic client
    if anthropic_client is None:
        try:
            anthropic_client = anthropic.Anthropic()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to initialize Anthropic client for token counting: {e}")
            # Return 0 if we can't create the client
            return 0

    count = 0
    try:
        for message in messages:
            count += anthropic_client.count_tokens(text=message)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Failed to count tokens with Anthropic client: {e}")
        return 0

    return count


def _set_ai21_span_attributes(span, request_body, response_body, metric_params):
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_OPERATION_NAME, GenAiOperationNameValues.TEXT_COMPLETION.value
    )
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
    for completion in response_body.get("completions"):
        output_messages.append({
            "role": "assistant",
            "content": completion.get("data").get("text"),
        })
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
        json.dumps(output_messages),
    )


def _set_llama_span_attributes(span, request_body, response_body, metric_params):
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_OPERATION_NAME, GenAiOperationNameValues.TEXT_COMPLETION.value
    )
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
        json.dumps([{"role": "user", "content": request_body.get("prompt")}]),
    )


def _set_llama_response_span_attributes(span, response_body):
    if response_body.get("generation"):
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps([{"role": "assistant", "content": response_body.get("generation")}]),
        )
    else:
        output_messages = []
        for generation in response_body.get("generations"):
            output_messages.append({"role": "assistant", "content": generation})
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps(output_messages),
        )


def _set_amazon_span_attributes(
    span, request_body, response_body, headers, metric_params
):
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_OPERATION_NAME, GenAiOperationNameValues.TEXT_COMPLETION.value
    )

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
            json.dumps([{"role": "user", "content": request_body.get("inputText")}]),
        )
    else:
        input_messages = []
        if "system" in request_body:
            for prompt in request_body["system"]:
                input_messages.append({
                    "role": "system",
                    "content": prompt.get("text"),
                })
        for prompt in request_body["messages"]:
            input_messages.append({
                "role": prompt.get("role"),
                "content": json.dumps(prompt.get("content", ""), default=str),
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
            output_messages.append({
                "role": "assistant",
                "content": result.get("outputText"),
            })
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps(output_messages),
        )
    elif "outputText" in response_body:
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps([{"role": "assistant", "content": response_body.get("outputText")}]),
        )
    elif "output" in response_body:
        msgs = response_body.get("output").get("message", {}).get("content", [])
        output_messages = []
        for msg in msgs:
            output_messages.append({
                "role": "assistant",
                "content": msg.get("text"),
            })
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps(output_messages),
        )


def _set_imported_model_span_attributes(
    span, request_body, response_body, metric_params
):
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_OPERATION_NAME, GenAiOperationNameValues.TEXT_COMPLETION.value
    )
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
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
        json.dumps([{"role": "assistant", "content": response_body.get("generation")}]),
    )


def _set_imported_model_prompt_span_attributes(span, request_body):
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_INPUT_MESSAGES,
        json.dumps([{"role": "user", "content": request_body.get("prompt")}]),
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
        "vendor": response_vendor,
        GenAIAttributes.GEN_AI_RESPONSE_MODEL: response_model,
        GenAIAttributes.GEN_AI_PROVIDER_NAME: GenAiSystemValues.AWS_BEDROCK.value,
        "stream": is_streaming,
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


def set_converse_input_prompt_span_attributes(kwargs, span):
    if not should_send_prompts():
        return
    input_messages = []
    if "system" in kwargs:
        for prompt in kwargs["system"]:
            input_messages.append({
                "role": "system",
                "content": prompt.get("text"),
            })
    if "messages" in kwargs:
        for prompt in kwargs["messages"]:
            input_messages.append({
                "role": prompt.get("role"),
                "content": json.dumps(prompt.get("content", ""), default=str),
            })
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_INPUT_MESSAGES,
        json.dumps(input_messages),
    )


def set_converse_response_span_attributes(response, span):
    if not should_send_prompts():
        return
    if "output" in response:
        message = response["output"]["message"]
        contents = [content.get("text") for content in message["content"]]
        content = contents[0] if len(contents) == 1 else json.dumps(contents)
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps([{"role": message.get("role"), "content": content}]),
        )


def set_converse_streaming_response_span_attributes(response, role, span):
    if not should_send_prompts():
        return
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
        json.dumps([{"role": role, "content": "".join(response)}]),
    )


def converse_usage_record(span, response, metric_params):
    prompt_tokens = 0
    completion_tokens = 0
    if "usage" in response:
        if "inputTokens" in response["usage"]:
            prompt_tokens = response["usage"]["inputTokens"]
        if "outputTokens" in response["usage"]:
            completion_tokens = response["usage"]["outputTokens"]

    _record_usage_to_span(
        span,
        prompt_tokens,
        completion_tokens,
        metric_params,
    )
