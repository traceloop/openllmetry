import json

from opentelemetry.instrumentation.llamaindex._message_utils import (
    build_completion_output_message,
    build_input_messages,
    build_output_message,
)
from opentelemetry.instrumentation.llamaindex._response_utils import (
    detect_provider_name,
    extract_finish_reasons,
    extract_model_from_raw,
    extract_response_id,
    extract_token_usage,
)
from opentelemetry.instrumentation.llamaindex.utils import (
    dont_throw,
    should_send_prompts,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)


@dont_throw
def set_llm_chat_request(event, span) -> None:
    if not span.is_recording():
        return

    if should_send_prompts():
        msgs = build_input_messages(event.messages)
        span.set_attribute(GenAIAttributes.GEN_AI_INPUT_MESSAGES, json.dumps(msgs))


@dont_throw
def set_llm_chat_request_model_attributes(event, span):
    if span and not span.is_recording():
        return

    model_dict = event.model_dict
    span.set_attribute(GenAIAttributes.GEN_AI_OPERATION_NAME, "chat")

    class_name = model_dict.get("class_name")
    provider = detect_provider_name(class_name)
    if provider:
        span.set_attribute(GenAIAttributes.GEN_AI_PROVIDER_NAME, provider)

    # For StructuredLLM, the model and temperature are nested under model_dict.llm
    if "llm" in model_dict:
        model_dict = model_dict.get("llm", {})

    span.set_attribute(GenAIAttributes.GEN_AI_REQUEST_MODEL, model_dict.get("model"))
    span.set_attribute(
        GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, model_dict.get("temperature")
    )


@dont_throw
def set_llm_chat_response(event, span) -> None:
    if not span.is_recording():
        return

    response = event.response
    finish_reasons = extract_finish_reasons(response.raw) if response.raw else []

    # finish_reasons is NOT gated by should_send_prompts() — it's metadata, not content
    if finish_reasons:
        span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)

    if should_send_prompts():
        fr = finish_reasons[0] if finish_reasons else None
        output_msg = build_output_message(response.message, finish_reason=fr)
        span.set_attribute(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, json.dumps([output_msg]))


@dont_throw
def set_llm_chat_response_model_attributes(event, span):
    if not span.is_recording():
        return

    response = event.response

    if not (raw := response.raw):
        return

    model = extract_model_from_raw(raw)
    if model:
        span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_MODEL, model)

    response_id = extract_response_id(raw)
    if response_id:
        span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_ID, response_id)

    usage = extract_token_usage(raw)
    if usage.output_tokens is not None:
        span.set_attribute(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, int(usage.output_tokens))
    if usage.input_tokens is not None:
        span.set_attribute(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, int(usage.input_tokens))
    if usage.total_tokens is not None:
        span.set_attribute(SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS, int(usage.total_tokens))

    # CRITICAL: finish_reasons is NOT gated by should_send_prompts()
    finish_reasons = extract_finish_reasons(raw)
    if finish_reasons:
        span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)


@dont_throw
def set_llm_predict_response(event, span) -> None:
    if should_send_prompts():
        output_msg = build_completion_output_message(event.output or "")
        span.set_attribute(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, json.dumps([output_msg]))


@dont_throw
def set_embedding(event, span) -> None:
    model_dict = event.model_dict
    span.set_attribute(GenAIAttributes.GEN_AI_OPERATION_NAME, "embeddings")
    span.set_attribute(GenAIAttributes.GEN_AI_REQUEST_MODEL, model_dict.get("model_name"))


@dont_throw
def set_rerank(event, span) -> None:
    if not span.is_recording():
        return
    if should_send_prompts():
        msg = [{"role": "user", "parts": [{"type": "text", "content": event.query.query_str}]}]
        span.set_attribute(GenAIAttributes.GEN_AI_INPUT_MESSAGES, json.dumps(msg))


@dont_throw
def set_rerank_model_attributes(event, span):
    if not span.is_recording():
        return
    span.set_attribute(GenAIAttributes.GEN_AI_OPERATION_NAME, "rerank")
    span.set_attribute(GenAIAttributes.GEN_AI_REQUEST_MODEL, event.model_name)
    span.set_attribute("rerank.top_n", event.top_n)


@dont_throw
def set_tool(event, span) -> None:
    span.set_attribute("tool.name", event.tool.name)
    span.set_attribute("tool.description", event.tool.description)
    span.set_attribute("tool.arguments", event.arguments)
