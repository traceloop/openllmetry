from opentelemetry.instrumentation.voyageai.utils import (
    dont_throw,
    dump_object,
    should_send_prompts,
    should_emit_events,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace.status import Status, StatusCode


# Operation name constants
OPERATION_EMBEDDINGS = "embeddings"
OPERATION_RERANK = "rerank"


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


@dont_throw
def set_input_content_attributes(span, operation_name, kwargs):
    if not span.is_recording():
        return

    if should_send_prompts() and not should_emit_events():
        if operation_name == OPERATION_EMBEDDINGS:
            # For embeddings, capture texts as input
            texts = kwargs.get("texts", [])
            if texts:
                _set_span_attribute(
                    span,
                    f"{GenAIAttributes.GEN_AI_PROMPT}.0.role",
                    "user",
                )
                # Store texts as JSON array
                _set_span_attribute(
                    span,
                    f"{GenAIAttributes.GEN_AI_PROMPT}.0.content",
                    dump_object([{"type": "text", "text": text} for text in texts]),
                )

        elif operation_name == OPERATION_RERANK:
            # For rerank, capture documents as system prompts and query as user prompt
            documents = kwargs.get("documents", [])
            for index, document in enumerate(documents):
                _set_span_attribute(
                    span, f"{GenAIAttributes.GEN_AI_PROMPT}.{index}.role", "system"
                )
                _set_span_attribute(
                    span, f"{GenAIAttributes.GEN_AI_PROMPT}.{index}.content", document
                )

            # Query is the user prompt
            _set_span_attribute(
                span,
                f"{GenAIAttributes.GEN_AI_PROMPT}.{len(documents)}.role",
                "user",
            )
            _set_span_attribute(
                span,
                f"{GenAIAttributes.GEN_AI_PROMPT}.{len(documents)}.content",
                kwargs.get("query"),
            )


@dont_throw
def set_response_content_attributes(span, operation_name, response):
    if not span.is_recording():
        return

    if should_send_prompts():
        if operation_name == OPERATION_RERANK:
            _set_span_rerank_response(span, response)
    span.set_status(Status(StatusCode.OK))


@dont_throw
def set_span_request_attributes(span, kwargs):
    if not span.is_recording():
        return

    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MODEL, kwargs.get("model"))
    # Voyage AI specific: top_k for rerank
    if kwargs.get("top_k"):
        _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_K, kwargs.get("top_k"))


@dont_throw
def set_span_response_attributes(span, operation_name, response):
    if not span.is_recording():
        return

    # Access total_tokens directly from the response object
    # Voyage AI SDK provides this as a direct attribute
    total_tokens = getattr(response, "total_tokens", None)
    if total_tokens:
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
            total_tokens,
        )

    # For embeddings, capture the dimension count
    if operation_name == OPERATION_EMBEDDINGS:
        embeddings = getattr(response, "embeddings", [])
        if embeddings and len(embeddings) > 0:
            # Get dimension from first embedding
            dimension = len(embeddings[0]) if embeddings[0] else 0
            _set_span_attribute(
                span,
                GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
                dimension,
            )


def _set_span_rerank_response(span, response):
    # Access results directly from response object
    results = getattr(response, "results", [])

    for idx, result in enumerate(results):
        prefix = f"{GenAIAttributes.GEN_AI_COMPLETION}.{idx}"
        _set_span_attribute(span, f"{prefix}.role", "assistant")

        # Access attributes directly from the result object
        index = getattr(result, "index", idx)
        relevance_score = getattr(result, "relevance_score", 0.0)
        document = getattr(result, "document", "")

        content = f"Doc {index}, Score: {relevance_score}"
        if document:
            content += f"\n{document}"
        _set_span_attribute(
            span,
            f"{prefix}.content",
            content,
        )
