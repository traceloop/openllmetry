"""Span attribute helpers for the OCI Generative AI instrumentation.

Request/response shapes handled here (all from ``oci.generative_ai_inference.models``):

* ``chat``: ``ChatDetails.chat_request`` is a ``GenericChatRequest`` (``messages`` list, OpenAI-like, used by
  Meta, OpenAI, Google and xAI models), a ``CohereChatRequest`` (``message`` + ``chat_history``) or a
  ``CohereChatRequestV2`` (``messages`` list). The result is a ``ChatResult`` whose ``chat_response`` is the
  matching ``GenericChatResponse`` / ``CohereChatResponse`` / ``CohereChatResponseV2``.
* ``generate_text``: legacy completion API with ``CohereLlmInferenceRequest`` / ``LlamaLlmInferenceRequest``.
* ``embed_text``: ``EmbedTextDetails`` -> ``EmbedTextResult``.
* ``rerank_text``: ``RerankTextDetails`` -> ``RerankTextResult``.
"""

import json
import logging

from opentelemetry.instrumentation.oci_genai.utils import (
    dont_throw,
    get_server_address,
    model_as_dict,
    set_span_attribute,
    should_send_prompts,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes.server_attributes import SERVER_ADDRESS
from opentelemetry.semconv_ai import SpanAttributes

logger = logging.getLogger(__name__)

OCI_GENAI_PROVIDER = "oracle_cloud.generative_ai"
"""``gen_ai.provider.name`` value for OCI Generative AI (open-telemetry/semantic-conventions-genai#526)."""

CHAT = GenAIAttributes.GenAiOperationNameValues.CHAT.value
TEXT_COMPLETION = GenAIAttributes.GenAiOperationNameValues.TEXT_COMPLETION.value
EMBEDDINGS = GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value
RERANK = "rerank"

# OCI specific attributes (vendor namespace, same pattern as gen_ai.openai.* / gen_ai.bedrock.*)
GEN_AI_OCI_SERVING_MODE = "gen_ai.oci.serving_mode"
GEN_AI_OCI_ENDPOINT_ID = "gen_ai.oci.endpoint_id"
GEN_AI_OCI_API_FORMAT = "gen_ai.oci.api_format"
GEN_AI_OCI_EMBEDDINGS_INPUT_COUNT = "gen_ai.oci.embeddings.input_count"
GEN_AI_OCI_EMBEDDINGS_INPUT_TYPE = "gen_ai.oci.embeddings.input_type"
GEN_AI_OCI_RERANK_TOP_N = "gen_ai.oci.rerank.top_n"
GEN_AI_OCI_RERANK_DOCUMENT_COUNT = "gen_ai.oci.rerank.document_count"

# OCI finish reason -> OTel GenAI finish reason (stop, length, content_filter, tool_call, error).
# Unknown values are passed through lower-cased.
FINISH_REASON_MAP = {
    # GENERIC api format (OpenAI compatible)
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_call",
    "content_filter": "content_filter",
    # COHERE / COHEREV2 api formats and legacy Cohere text generation
    "COMPLETE": "stop",
    "STOP_SEQUENCE": "stop",
    "MAX_TOKENS": "length",
    "TOOL_CALL": "tool_call",
    "ERROR": "error",
    "ERROR_LIMIT": "error",
    "ERROR_TOXIC": "content_filter",
}

_DETAILS_REQUEST_ATTRIBUTE = {CHAT: "chat_request", TEXT_COMPLETION: "inference_request"}


def map_finish_reason(reason):
    if not reason:
        return ""
    reason = str(reason)
    return FINISH_REASON_MAP.get(reason, reason.lower())


def _normalize_role(role):
    if not role:
        return "user"
    role = str(role).lower()
    return "assistant" if role == "chatbot" else role


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


# ---------------------------------------------------------------------------
# OTel message parts
# ---------------------------------------------------------------------------


def _text_part(content):
    return {"type": "text", "content": content}


def _reasoning_part(content):
    return {"type": "reasoning", "content": content}


def _image_part(url):
    if isinstance(url, str) and url.startswith("data:"):
        try:
            header, data = url.split(",", 1)
            mime_type = header.split(":", 1)[1].split(";", 1)[0]
            return {"type": "blob", "modality": "image", "mime_type": mime_type, "content": data}
        except (ValueError, IndexError):
            pass
    return {"type": "uri", "modality": "image", "uri": url or ""}


def _parse_arguments(raw):
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return raw
    if raw is None:
        return {}
    return model_as_dict(raw)


def _tool_call_part(name, arguments=None, tool_id=None):
    part = {"type": "tool_call", "name": name or "", "arguments": _parse_arguments(arguments)}
    if tool_id:
        part["id"] = tool_id
    return part


def _tool_call_response_part(tool_id, response):
    return {
        "type": "tool_call_response",
        "id": tool_id or "",
        "response": response if response is not None else "",
    }


def _output_message(role, parts, finish_reason):
    """OutputMessage per the OTel GenAI JSON schema: role, parts and finish_reason are required."""
    return {"role": role, "parts": parts, "finish_reason": finish_reason or ""}


def _parts_text(parts):
    return "".join(part.get("content", "") for part in parts if part.get("type") == "text")


def _content_to_parts(content):
    """Convert OCI ``ChatContent`` / ``CohereContentV2`` lists (or a plain string) to OTel parts."""
    if content is None:
        return []
    if isinstance(content, str):
        return [_text_part(content)] if content else []
    parts = []
    for item in content:
        if isinstance(item, str):
            if item:
                parts.append(_text_part(item))
            continue
        item_type = str(getattr(item, "type", "") or "").upper()
        if item_type == "TEXT" or (not item_type and hasattr(item, "text")):
            text = getattr(item, "text", None)
            if text:
                parts.append(_text_part(text))
        elif item_type == "THINKING":
            thinking = getattr(item, "thinking", None)
            if thinking:
                parts.append(_reasoning_part(thinking))
        elif item_type == "IMAGE":
            parts.append(_image_part(getattr(getattr(item, "image_url", None), "url", None)))
        else:
            # Stopgap for audio/video/document content: encode as text to keep the parts array valid.
            parts.append(_text_part(json.dumps(model_as_dict(item), default=str)))
    return parts


def _field(obj, name):
    """Read ``name`` from an SDK model or from a plain dict (some SDK fields are untyped ``object``)."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _tool_calls_to_parts(tool_calls):
    """Convert ``FunctionCall`` (GENERIC), ``CohereToolCallV2`` or ``CohereToolCall`` lists to OTel parts."""
    parts = []
    for tool_call in tool_calls or []:
        function = _field(tool_call, "function")
        if function is not None:
            # CohereToolCallV2: {"id", "type", "function": {"name", "arguments"}}
            parts.append(
                _tool_call_part(_field(function, "name"), _field(function, "arguments"), _field(tool_call, "id"))
            )
        elif _field(tool_call, "parameters") is not None:
            # CohereToolCall (v1): {"name", "parameters"}
            parts.append(_tool_call_part(_field(tool_call, "name"), _field(tool_call, "parameters")))
        else:
            # FunctionCall (GENERIC): {"id", "type", "name", "arguments"}
            parts.append(
                _tool_call_part(_field(tool_call, "name"), _field(tool_call, "arguments"), _field(tool_call, "id"))
            )
    return parts


def _cohere_tool_results_to_parts(tool_results):
    parts = []
    for result in tool_results or []:
        call = getattr(result, "call", None)
        parts.append(
            _tool_call_response_part(
                getattr(call, "name", None),
                model_as_dict(getattr(result, "outputs", None)) or "",
            )
        )
    return parts


def _message_to_otel(message):
    """Convert a GENERIC ``Message`` or a ``CohereMessageV2`` into an OTel input/output message dict."""
    role = _normalize_role(getattr(message, "role", None))
    content_parts = _content_to_parts(getattr(message, "content", None))
    if role == "tool":
        parts = [_tool_call_response_part(getattr(message, "tool_call_id", None), _parts_text(content_parts))]
    else:
        parts = content_parts
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            parts.append(_reasoning_part(reasoning))
        parts.extend(_tool_calls_to_parts(getattr(message, "tool_calls", None)))
    return {"role": role, "parts": parts}


def _cohere_history_to_messages(chat_history):
    messages = []
    for item in chat_history or []:
        role = _normalize_role(getattr(item, "role", None))
        if role == "tool":
            parts = _cohere_tool_results_to_parts(getattr(item, "tool_results", None))
        else:
            parts = []
            text = getattr(item, "message", None)
            if text:
                parts.append(_text_part(text))
            parts.extend(_tool_calls_to_parts(getattr(item, "tool_calls", None)))
        messages.append({"role": role, "parts": parts})
    return messages


def chat_request_to_messages(request):
    """Return ``(input_messages, system_instruction_parts)`` for any OCI chat request flavour."""
    if request is None:
        return [], []
    api_format = str(getattr(request, "api_format", "") or "").upper()
    if api_format == "COHERE":
        system_parts = []
        preamble = getattr(request, "preamble_override", None)
        if preamble:
            system_parts.append(_text_part(preamble))
        messages = _cohere_history_to_messages(getattr(request, "chat_history", None))
        message = getattr(request, "message", None)
        if message:
            messages.append({"role": "user", "parts": [_text_part(message)]})
        tool_results = getattr(request, "tool_results", None)
        if tool_results:
            messages.append({"role": "tool", "parts": _cohere_tool_results_to_parts(tool_results)})
        return messages, system_parts

    # GENERIC and COHEREV2 both carry a ``messages`` list
    return [_message_to_otel(message) for message in getattr(request, "messages", None) or []], []


def chat_response_to_messages(chat_response):
    """Return OTel output messages for a ``GenericChatResponse`` / ``CohereChatResponse`` / ``CohereChatResponseV2``."""
    if chat_response is None:
        return []
    api_format = str(getattr(chat_response, "api_format", "") or "").upper()
    finish_reason = map_finish_reason(getattr(chat_response, "finish_reason", None))
    if api_format == "COHERE":
        parts = []
        text = getattr(chat_response, "text", None)
        if text:
            parts.append(_text_part(text))
        parts.extend(_tool_calls_to_parts(getattr(chat_response, "tool_calls", None)))
        return [_output_message("assistant", parts, finish_reason)]
    if api_format == "COHEREV2":
        message = getattr(chat_response, "message", None)
        otel = _message_to_otel(message) if message is not None else {"role": "assistant", "parts": []}
        return [_output_message(otel["role"], otel["parts"], finish_reason)]

    messages = []
    for choice in getattr(chat_response, "choices", None) or []:
        message = getattr(choice, "message", None)
        otel = _message_to_otel(message) if message is not None else {"role": "assistant", "parts": []}
        messages.append(
            _output_message(otel["role"], otel["parts"], map_finish_reason(getattr(choice, "finish_reason", None)))
        )
    return messages


def text_completion_response_to_messages(inference_response):
    """Return OTel output messages for a legacy ``CohereLlmInferenceResponse`` / ``LlamaLlmInferenceResponse``."""
    messages = []
    if inference_response is None:
        return messages
    for generated in getattr(inference_response, "generated_texts", None) or []:
        messages.append(
            _output_message(
                "assistant",
                [_text_part(getattr(generated, "text", None) or "")],
                map_finish_reason(getattr(generated, "finish_reason", None)),
            )
        )
    for choice in getattr(inference_response, "choices", None) or []:
        messages.append(
            _output_message(
                "assistant",
                [_text_part(getattr(choice, "text", None) or "")],
                map_finish_reason(getattr(choice, "finish_reason", None)),
            )
        )
    return messages


def rerank_response_to_messages(result):
    messages = []
    for rank in getattr(result, "document_ranks", None) or []:
        content = f"Doc {getattr(rank, 'index', None)}, Score: {getattr(rank, 'relevance_score', None)}"
        document = getattr(rank, "document", None)
        if document:
            if not isinstance(document, str):
                document = json.dumps(model_as_dict(document), default=str)
            content += "\n" + document
        messages.append(_output_message("assistant", [_text_part(content)], "stop"))
    return messages


def stream_choices_to_messages(choices):
    """Return OTel output messages for the choices aggregated by ``StreamAccumulator``."""
    messages = []
    for choice in choices:
        parts = []
        if choice.get("text"):
            parts.append(_text_part(choice["text"]))
        if choice.get("reasoning"):
            parts.append(_reasoning_part(choice["reasoning"]))
        for tool_call in choice.get("tool_calls") or []:
            parts.append(_tool_call_part(tool_call.get("name"), tool_call.get("arguments"), tool_call.get("id")))
        messages.append(_output_message("assistant", parts, map_finish_reason(choice.get("finish_reason"))))
    return messages


def response_to_output_messages(operation, data):
    if data is None:
        return []
    if operation == CHAT:
        return chat_response_to_messages(getattr(data, "chat_response", None))
    if operation == TEXT_COMPLETION:
        return text_completion_response_to_messages(getattr(data, "inference_response", None))
    if operation == RERANK:
        return rerank_response_to_messages(data)
    return []


def embedding_input_parts(details):
    """Return one OTel parts list per embedding input (``inputs`` strings or ``embed_contents`` items)."""
    inputs = getattr(details, "inputs", None)
    if inputs:
        return [[_text_part(text)] for text in inputs]
    parts = []
    for content in getattr(details, "embed_contents", None) or []:
        content_type = str(getattr(content, "type", "") or "").upper()
        if content_type == "IMAGE":
            parts.append([_image_part(getattr(getattr(content, "image_url", None), "url", None))])
        else:
            parts.append([_text_part(getattr(content, "text", None) or "")])
    return parts


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


def get_request_model(details):
    """Model id for on-demand serving, endpoint id for dedicated AI cluster endpoints."""
    serving_mode = getattr(details, "serving_mode", None)
    return (
        getattr(serving_mode, "model_id", None)
        or getattr(serving_mode, "endpoint_id", None)
        or "unknown"
    )


def get_inner_request(operation, details):
    attribute = _DETAILS_REQUEST_ATTRIBUTE.get(operation)
    return getattr(details, attribute, None) if attribute else details


def is_stream_request(operation, details):
    return bool(getattr(get_inner_request(operation, details), "is_stream", False))


def usage_to_dict(usage, include_output=True):
    """Normalise an OCI ``Usage`` model or a camelCase streaming ``usage`` payload."""
    if usage is None:
        return None
    if isinstance(usage, dict):
        prompt_details = usage.get("promptTokensDetails") or usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completionTokensDetails") or usage.get("completion_tokens_details") or {}
        data = {
            "input_tokens": _first_not_none(usage.get("promptTokens"), usage.get("prompt_tokens")),
            "output_tokens": _first_not_none(usage.get("completionTokens"), usage.get("completion_tokens")),
            "total_tokens": _first_not_none(usage.get("totalTokens"), usage.get("total_tokens")),
            "cached_tokens": _first_not_none(prompt_details.get("cachedTokens"), prompt_details.get("cached_tokens"))
            if isinstance(prompt_details, dict)
            else None,
            "reasoning_tokens": _first_not_none(
                completion_details.get("reasoningTokens"), completion_details.get("reasoning_tokens")
            )
            if isinstance(completion_details, dict)
            else None,
        }
    else:
        data = {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "cached_tokens": getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None),
            "reasoning_tokens": getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None),
        }
    if not include_output:
        data["output_tokens"] = None
        data["reasoning_tokens"] = None
    if data["total_tokens"] is None and data["input_tokens"] is not None and data["output_tokens"] is not None:
        data["total_tokens"] = data["input_tokens"] + data["output_tokens"]
    return data


def set_usage_attributes(span, usage):
    if not usage:
        return
    set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, usage.get("input_tokens"))
    set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, usage.get("output_tokens"))
    set_span_attribute(span, SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS, usage.get("total_tokens"))
    set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, usage.get("cached_tokens"))
    set_span_attribute(span, SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS, usage.get("reasoning_tokens"))


def record_usage_metrics(token_histogram, usage, attributes):
    if token_histogram is None or not usage:
        return
    for token_type, key in (("input", "input_tokens"), ("output", "output_tokens")):
        value = usage.get(key)
        if isinstance(value, int) and value >= 0:
            token_histogram.record(
                value,
                attributes={**attributes, GenAIAttributes.GEN_AI_TOKEN_TYPE: token_type},
            )


def _set_finish_reasons(span, messages):
    finish_reasons = [message["finish_reason"] for message in messages if message.get("finish_reason")]
    if finish_reasons:
        set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)


# ---------------------------------------------------------------------------
# Span attribute setters
# ---------------------------------------------------------------------------


@dont_throw
def set_request_attributes(span, operation, details, instance):
    """Request metadata: serving mode, endpoint, sampling parameters and tool definitions."""
    if not span.is_recording():
        return

    serving_mode = getattr(details, "serving_mode", None)
    set_span_attribute(span, GEN_AI_OCI_SERVING_MODE, getattr(serving_mode, "serving_type", None))
    set_span_attribute(span, GEN_AI_OCI_ENDPOINT_ID, getattr(serving_mode, "endpoint_id", None))
    set_span_attribute(span, SERVER_ADDRESS, get_server_address(instance))

    request = get_inner_request(operation, details)
    if operation in (CHAT, TEXT_COMPLETION):
        set_span_attribute(
            span,
            GEN_AI_OCI_API_FORMAT,
            _first_not_none(getattr(request, "api_format", None), getattr(request, "runtime_type", None)),
        )
        set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, getattr(request, "temperature", None))
        set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_P, getattr(request, "top_p", None))
        set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_K, getattr(request, "top_k", None))
        set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS,
            _first_not_none(getattr(request, "max_tokens", None), getattr(request, "max_completion_tokens", None)),
        )
        set_span_attribute(
            span, GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY, getattr(request, "frequency_penalty", None)
        )
        set_span_attribute(
            span, GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY, getattr(request, "presence_penalty", None)
        )
        set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_SEED, getattr(request, "seed", None))
        stop = _first_not_none(getattr(request, "stop", None), getattr(request, "stop_sequences", None))
        if stop:
            set_span_attribute(
                span,
                GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES,
                list(stop) if isinstance(stop, (list, tuple)) else [stop],
            )
        set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_CHOICE_COUNT, getattr(request, "num_generations", None))
        set_span_attribute(span, SpanAttributes.GEN_AI_IS_STREAMING, bool(getattr(request, "is_stream", False)))
        tools = getattr(request, "tools", None)
        if tools and should_send_prompts():
            set_span_attribute(
                span, GenAIAttributes.GEN_AI_TOOL_DEFINITIONS, json.dumps(model_as_dict(tools), default=str)
            )
    elif operation == EMBEDDINGS:
        set_span_attribute(span, GEN_AI_OCI_EMBEDDINGS_INPUT_COUNT, len(embedding_input_parts(details)))
        set_span_attribute(span, GEN_AI_OCI_EMBEDDINGS_INPUT_TYPE, getattr(details, "input_type", None))
        embedding_types = getattr(details, "embedding_types", None)
        if embedding_types:
            set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_ENCODING_FORMATS, list(embedding_types))
        set_span_attribute(
            span, GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT, getattr(details, "output_dimensions", None)
        )
    elif operation == RERANK:
        set_span_attribute(span, GEN_AI_OCI_RERANK_TOP_N, getattr(details, "top_n", None))
        set_span_attribute(span, GEN_AI_OCI_RERANK_DOCUMENT_COUNT, len(getattr(details, "documents", None) or []))


@dont_throw
def set_input_attributes(span, operation, details):
    """Prompt content (``gen_ai.input.messages`` / ``gen_ai.system_instructions``); gated by TRACELOOP_TRACE_CONTENT."""
    if not span.is_recording() or not should_send_prompts():
        return

    messages = []
    if operation == CHAT:
        messages, system_parts = chat_request_to_messages(getattr(details, "chat_request", None))
        if system_parts:
            set_span_attribute(span, GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS, json.dumps(system_parts))
    elif operation == TEXT_COMPLETION:
        prompt = getattr(getattr(details, "inference_request", None), "prompt", None)
        if prompt:
            messages = [{"role": "user", "parts": [_text_part(prompt)]}]
    elif operation == EMBEDDINGS:
        messages = [{"role": "user", "parts": parts} for parts in embedding_input_parts(details)]
    elif operation == RERANK:
        messages = [
            {"role": "system", "parts": [_text_part(document if isinstance(document, str) else str(document))]}
            for document in getattr(details, "documents", None) or []
        ]
        query = getattr(details, "input", None)
        if query:
            messages.append({"role": "user", "parts": [_text_part(query)]})

    if messages:
        set_span_attribute(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES, json.dumps(messages))


@dont_throw
def set_response_attributes(span, operation, response):
    """Response metadata (model, id, finish reasons, usage). Returns the normalised usage for metrics."""
    if not span.is_recording():
        return None
    data = getattr(response, "data", None)
    if data is None:
        return None

    set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, getattr(data, "model_id", None))

    if operation == CHAT:
        chat_response = getattr(data, "chat_response", None)
        set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, getattr(chat_response, "id", None))
        _set_finish_reasons(span, chat_response_to_messages(chat_response))
        usage = usage_to_dict(getattr(chat_response, "usage", None))
        set_usage_attributes(span, usage)
        return usage

    if operation == TEXT_COMPLETION:
        _set_finish_reasons(span, text_completion_response_to_messages(getattr(data, "inference_response", None)))
        return None

    if operation == EMBEDDINGS:
        set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, getattr(data, "id", None))
        set_span_attribute(span, GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT, _embedding_dimensions(data))
        usage = usage_to_dict(getattr(data, "usage", None), include_output=False)
        set_usage_attributes(span, usage)
        return usage

    if operation == RERANK:
        set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, getattr(data, "id", None))
    return None


def _embedding_dimensions(result):
    embeddings = getattr(result, "embeddings", None)
    if not embeddings:
        by_type = getattr(result, "embeddings_by_type", None)
        if isinstance(by_type, dict):
            embeddings = next((vectors for vectors in by_type.values() if vectors), None)
    if embeddings and hasattr(embeddings[0], "__len__"):
        return len(embeddings[0])
    return None


@dont_throw
def set_output_attributes(span, operation, response):
    """Completion content (``gen_ai.output.messages``); gated by TRACELOOP_TRACE_CONTENT."""
    if not span.is_recording() or not should_send_prompts():
        return
    messages = response_to_output_messages(operation, getattr(response, "data", None))
    if messages:
        set_span_attribute(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, json.dumps(messages))


@dont_throw
def set_streaming_response_attributes(span, accumulator):
    """Response metadata aggregated from a consumed SSE stream. Returns the normalised usage for metrics."""
    if not span.is_recording():
        return None
    set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, accumulator.response_id)
    _set_finish_reasons(span, stream_choices_to_messages(accumulator.choices_list()))
    usage = usage_to_dict(accumulator.usage)
    set_usage_attributes(span, usage)
    return usage


@dont_throw
def set_streaming_output_attributes(span, accumulator):
    if not span.is_recording() or not should_send_prompts():
        return
    messages = stream_choices_to_messages(accumulator.choices_list())
    if messages:
        set_span_attribute(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, json.dumps(messages))
