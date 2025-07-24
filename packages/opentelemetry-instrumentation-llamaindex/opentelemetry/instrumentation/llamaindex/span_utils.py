from llama_index.core.base.llms.types import MessageRole
from opentelemetry.instrumentation.llamaindex.utils import (
    dont_throw,
    should_send_prompts,
)
from opentelemetry.semconv_ai import (
    LLMRequestTypeValues,
    SpanAttributes,
)


@dont_throw
def set_llm_chat_request(event, span) -> None:
    if not span.is_recording():
        return

    if should_send_prompts():
        for idx, message in enumerate(event.messages):
            span.set_attribute(
                f"{SpanAttributes.LLM_PROMPTS}.{idx}.role", message.role.value
            )
            span.set_attribute(
                f"{SpanAttributes.LLM_PROMPTS}.{idx}.content", message.content
            )


@dont_throw
def set_llm_chat_request_model_attributes(event, span):
    if span and not span.is_recording():
        return

    model_dict = event.model_dict
    span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.CHAT.value)

    # For StructuredLLM, the model and temperature are nested under model_dict.llm
    if "llm" in model_dict:
        model_dict = model_dict.get("llm", {})

    span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model_dict.get("model"))
    span.set_attribute(
        SpanAttributes.LLM_REQUEST_TEMPERATURE, model_dict.get("temperature")
    )


@dont_throw
def set_llm_chat_response(event, span) -> None:
    if not span.is_recording():
        return

    response = event.response
    if should_send_prompts():
        for idx, message in enumerate(event.messages):
            span.set_attribute(
                f"{SpanAttributes.LLM_PROMPTS}.{idx}.role", message.role.value
            )
            span.set_attribute(
                f"{SpanAttributes.LLM_PROMPTS}.{idx}.content", message.content
            )
        span.set_attribute(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.role",
            response.message.role.value,
        )
        span.set_attribute(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
            response.message.content,
        )


@dont_throw
def set_llm_chat_response_model_attributes(event, span):
    if not span.is_recording():
        return

    response = event.response

    if not (raw := response.raw):
        return

    span.set_attribute(
        SpanAttributes.LLM_RESPONSE_MODEL,
        (
            raw.get("model") if "model" in raw else raw.model
        ),  # raw can be Any, not just ChatCompletion
    )
    if usage := raw.get("usage") if "usage" in raw else raw.usage:
        span.set_attribute(
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, usage.completion_tokens
        )
        span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.prompt_tokens)
        span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.total_tokens)
    if choices := raw.choices:
        span.set_attribute(
            SpanAttributes.LLM_RESPONSE_FINISH_REASON, choices[0].finish_reason
        )


@dont_throw
def set_llm_predict_response(event, span) -> None:
    if should_send_prompts():
        span.set_attribute(
            f"{SpanAttributes.LLM_COMPLETIONS}.role",
            MessageRole.ASSISTANT.value,
        )
        span.set_attribute(
            f"{SpanAttributes.LLM_COMPLETIONS}.content",
            event.output,
        )


@dont_throw
def set_embedding(event, span) -> None:
    model_dict = event.model_dict
    span.set_attribute(
        f"{LLMRequestTypeValues.EMBEDDING.value}.model_name",
        model_dict.get("model_name"),
    )


@dont_throw
def set_rerank(event, span) -> None:
    if not span.is_recording():
        return
    if should_send_prompts():
        span.set_attribute(
            f"{LLMRequestTypeValues.RERANK.value}.query",
            event.query.query_str,
        )


@dont_throw
def set_rerank_model_attributes(event, span):
    if not span.is_recording():
        return
    span.set_attribute(
        f"{LLMRequestTypeValues.RERANK.value}.model_name",
        event.model_name,
    )
    span.set_attribute(
        f"{LLMRequestTypeValues.RERANK.value}.top_n",
        event.top_n,
    )


@dont_throw
def set_tool(event, span) -> None:
    span.set_attribute("tool.name", event.tool.name)
    span.set_attribute("tool.description", event.tool.description)
    span.set_attribute("tool.arguments", event.arguments)
