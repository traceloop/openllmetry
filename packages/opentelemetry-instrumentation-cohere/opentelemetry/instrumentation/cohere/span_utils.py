from opentelemetry.instrumentation.cohere.utils import (
    dont_throw,
    dump_object,
    should_send_prompts,
    to_dict,
    should_emit_events,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import (
    LLMRequestTypeValues,
    SpanAttributes,
)
from opentelemetry.trace.status import Status, StatusCode


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


@dont_throw
def set_input_content_attributes(span, llm_request_type, kwargs):
    if not span.is_recording():
        return

    if should_send_prompts() and not should_emit_events():
        if llm_request_type == LLMRequestTypeValues.COMPLETION:
            _set_span_attribute(span, f"{GenAIAttributes.GEN_AI_PROMPT}.0.role", "user")
            _set_span_attribute(
                span, f"{GenAIAttributes.GEN_AI_PROMPT}.0.content", kwargs.get("prompt")
            )
        # client V1
        elif llm_request_type == LLMRequestTypeValues.CHAT and kwargs.get("message"):
            user_message_index = 0
            if system_message := kwargs.get("preamble"):
                _set_span_attribute(span, f"{GenAIAttributes.GEN_AI_PROMPT}.0.role", "system")
                _set_span_attribute(
                    span, f"{GenAIAttributes.GEN_AI_PROMPT}.0.content", system_message
                )
                user_message_index = 1
            _set_span_attribute(span, f"{GenAIAttributes.GEN_AI_PROMPT}.{user_message_index}.role", "user")
            _set_span_attribute(
                span, f"{GenAIAttributes.GEN_AI_PROMPT}.{user_message_index}.content", kwargs.get("message")
            )
        # client V2
        elif llm_request_type == LLMRequestTypeValues.CHAT and kwargs.get("messages"):
            for index, message in enumerate(kwargs.get("messages")):
                message_dict = to_dict(message)
                _set_span_attribute(span, f"{GenAIAttributes.GEN_AI_PROMPT}.{index}.role", message_dict.get("role"))
                _set_span_attribute(
                    span, f"{GenAIAttributes.GEN_AI_PROMPT}.{index}.content", message_dict.get("content")
                )

        if kwargs.get("tools"):
            for index, tool in enumerate(kwargs.get("tools")):
                function = tool.get("function")
                if not function:
                    continue
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{index}.name",
                    function.get("name"),
                )
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{index}.description",
                    function.get("description"),
                )
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{index}.parameters",
                    dump_object(function.get("parameters")),
                )
        elif llm_request_type == LLMRequestTypeValues.RERANK:
            for index, document in enumerate(kwargs.get("documents", [])):
                _set_span_attribute(
                    span, f"{GenAIAttributes.GEN_AI_PROMPT}.{index}.role", "system"
                )
                _set_span_attribute(
                    span, f"{GenAIAttributes.GEN_AI_PROMPT}.{index}.content", document
                )

            _set_span_attribute(
                span,
                f"{GenAIAttributes.GEN_AI_PROMPT}.{len(kwargs.get('documents'))}.role",
                "user",
            )
            _set_span_attribute(
                span,
                f"{GenAIAttributes.GEN_AI_PROMPT}.{len(kwargs.get('documents'))}.content",
                kwargs.get("query"),
            )
        elif llm_request_type == LLMRequestTypeValues.EMBEDDING:
            _set_span_attribute(
                span,
                f"{GenAIAttributes.GEN_AI_PROMPT}.0.role",
                "user",
            )
            inputs = kwargs.get("inputs")
            if not inputs:
                texts = kwargs.get("texts")
                inputs = [
                    {"type": "text", "text": text} for text in texts
                ]
            _set_span_attribute(
                span,
                f"{GenAIAttributes.GEN_AI_PROMPT}.0.content",
                dump_object(inputs),
            )


@dont_throw
def set_response_content_attributes(span, llm_request_type, response):
    if not span.is_recording():
        return

    if should_send_prompts():
        if llm_request_type == LLMRequestTypeValues.CHAT:
            _set_span_chat_response(span, response)
        elif llm_request_type == LLMRequestTypeValues.COMPLETION:
            _set_span_generations_response(span, response)
        elif llm_request_type == LLMRequestTypeValues.RERANK:
            _set_span_rerank_response(span, response)
    span.set_status(Status(StatusCode.OK))


@dont_throw
def set_span_request_attributes(span, kwargs):
    if not span.is_recording():
        return

    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MODEL, kwargs.get("model"))
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS, kwargs.get("max_tokens_to_sample")
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, kwargs.get("temperature")
    )
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_P, kwargs.get("p", kwargs.get("top_p")))
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_K, kwargs.get("k", kwargs.get("top_k")))

    if stop_sequences := kwargs.get("stop_sequences", []):
        _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES, dump_object(stop_sequences))

    # TODO: Migrate to GEN_AI_REQUEST_FREQUENCY_PENALTY and GEN_AI_REQUEST_PRESENCE_PENALTY
    _set_span_attribute(
        span, SpanAttributes.LLM_FREQUENCY_PENALTY, kwargs.get("frequency_penalty")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_PRESENCE_PENALTY, kwargs.get("presence_penalty")
    )


@dont_throw
def set_span_response_attributes(span, response):
    if not span.is_recording():
        return

    response_dict = to_dict(response)
    # Cohere API v1
    if (response_dict.get("response_id")):
        _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, response_dict.get("response_id"))
    # Cohere API v2
    elif (response_dict.get("id")):
        _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, response_dict.get("id"))

    # Cohere v4
    if token_count := response_dict.get("token_count"):
        token_count_dict = to_dict(token_count)
        input_tokens = token_count_dict.get("prompt_tokens", 0)
        output_tokens = token_count_dict.get("response_tokens", 0)
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

    # Cohere v5
    if response_dict.get("meta"):
        meta_dict = to_dict(response_dict.get("meta", {}))
        billed_units = meta_dict.get("billed_units", {})
        billed_units_dict = to_dict(billed_units)
        input_tokens = billed_units_dict.get("input_tokens", 0)
        output_tokens = billed_units_dict.get("output_tokens", 0)

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

    # Cohere API v2
    if response_dict.get("usage"):
        # usage also has usage.tokens of type UsageTokens. This usually
        # has the same number of output tokens, but many more input tokens
        # (possibly pre-prompted)")
        usage_dict = to_dict(response_dict.get("usage", {}))
        billed_units_dict = to_dict(usage_dict.get("billed_units", {}))
        input_tokens = billed_units_dict.get("input_tokens", 0)
        output_tokens = billed_units_dict.get("output_tokens", 0)
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


def _set_span_chat_response(span, response):
    index = 0
    prefix = f"{GenAIAttributes.GEN_AI_COMPLETION}.{index}"
    _set_span_attribute(span, f"{prefix}.role", "assistant")

    response_dict = to_dict(response)

    if finish_reason := response_dict.get("finish_reason"):
        _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason])

    # Cohere API v1
    if text := response_dict.get("text"):
        _set_span_attribute(span, f"{prefix}.content", text)
    # Cohere API v2
    elif message := response_dict.get("message"):
        message_dict = to_dict(message)
        content = message_dict.get("content") or []
        if tool_plan := message_dict.get("tool_plan"):
            content.append({
                "type": "text",
                "text": tool_plan,
            })
        # TODO: Add citations, similarly to tool_plan
        _set_span_attribute(span, f"{prefix}.content", dump_object(content))
        if tool_calls := message_dict.get("tool_calls"):
            tool_call_index = 0
            for tool_call in tool_calls:
                if not tool_call.get("function"):
                    continue
                function = tool_call.get("function")
                if tool_call.get("id"):
                    _set_span_attribute(span, f"{prefix}.tool_calls.{tool_call_index}.id", tool_call.get("id"))
                if function.get("name"):
                    _set_span_attribute(span, f"{prefix}.tool_calls.{tool_call_index}.name", function.get("name"))
                if function.get("arguments"):
                    # no dump_object here, since it's already a string (OpenAI-like)
                    _set_span_attribute(
                        span,
                        f"{prefix}.tool_calls.{tool_call_index}.arguments",
                        function.get("arguments"),
                    )
                tool_call_index += 1


def _set_span_generations_response(span, response):
    _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, response.id)
    if hasattr(response, "generations"):
        generations = response.generations  # Cohere v5
    else:
        generations = response  # Cohere v4

    for index, generation in enumerate(generations):
        prefix = f"{GenAIAttributes.GEN_AI_COMPLETION}.{index}"
        _set_span_attribute(span, f"{prefix}.content", generation.text)
        _set_span_attribute(span, f"gen_ai.response.{index}.id", generation.id)


def _set_span_rerank_response(span, response):
    _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, response.id)
    for idx, doc in enumerate(response.results):
        prefix = f"{GenAIAttributes.GEN_AI_COMPLETION}.{idx}"
        _set_span_attribute(span, f"{prefix}.role", "assistant")
        content = f"Doc {doc.index}, Score: {doc.relevance_score}"
        if hasattr(doc, "document") and doc.document:
            if hasattr(doc.document, "text"):
                content += f"\n{doc.document.text}"
            else:
                content += f"\n{doc.document.get('text')}"
        _set_span_attribute(
            span,
            f"{prefix}.content",
            content,
        )
