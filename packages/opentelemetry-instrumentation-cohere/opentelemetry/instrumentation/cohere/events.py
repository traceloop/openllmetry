"""Event utility functions for Cohere instrumentation."""

from typing import Any, Dict, List, Optional

from opentelemetry.semconv_ai import (
    LLMRequestTypeValues,
    SpanAttributes,
)

def create_prompt_event(
    content: str,
    role: Optional[str] = None,
    content_type: Optional[str] = None,
    model: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a prompt event for Cohere."""
    attributes = {
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.PROMPT,
        SpanAttributes.LLM_PROMPT_TEMPLATE: content,
        SpanAttributes.LLM_SYSTEM: "Cohere",
    }

    if role is not None:
        attributes[SpanAttributes.LLM_REQUEST_ROLE] = role

    if content_type is not None:
        attributes[SpanAttributes.LLM_REQUEST_CONTENT_TYPE] = content_type

    if model is not None:
        attributes[SpanAttributes.LLM_REQUEST_MODEL] = model

    if prompt_tokens is not None:
        attributes[SpanAttributes.LLM_TOKEN_COUNT] = prompt_tokens

    return {
        "name": "prompt",
        "attributes": attributes,
    }

def create_completion_event(
    completion: str,
    model: Optional[str] = None,
    completion_tokens: Optional[int] = None,
    role: Optional[str] = None,
    content_type: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a completion event for Cohere."""
    attributes = {
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION,
        SpanAttributes.LLM_COMPLETION: completion,
        SpanAttributes.LLM_SYSTEM: "Cohere",
    }

    if model is not None:
        attributes[SpanAttributes.LLM_RESPONSE_MODEL] = model

    if completion_tokens is not None:
        attributes[SpanAttributes.LLM_TOKEN_COUNT] = completion_tokens

    if role is not None:
        attributes[SpanAttributes.LLM_RESPONSE_ROLE] = role

    if content_type is not None:
        attributes[SpanAttributes.LLM_RESPONSE_CONTENT_TYPE] = content_type

    if finish_reason is not None:
        attributes[SpanAttributes.LLM_RESPONSE_FINISH_REASON] = finish_reason

    return {
        "name": "completion",
        "attributes": attributes,
    }

def create_rerank_event(
    documents: List[str],
    query: str,
    model: Optional[str] = None,
    scores: Optional[List[float]] = None,
    indices: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Create rerank events for Cohere."""
    events = []

    # Add documents as system prompts
    for idx, doc in enumerate(documents):
        events.append(create_prompt_event(
            content=doc,
            role="system",
            model=model,
        ))

    # Add query as user prompt
    events.append(create_prompt_event(
        content=query,
        role="user",
        model=model,
    ))

    # Add rerank results as completions
    if scores is not None and indices is not None:
        for idx, (score, doc_idx) in enumerate(zip(scores, indices)):
            content = f"Doc {doc_idx}, Score: {score}"
            if idx < len(documents):
                content += f"\n{documents[doc_idx]}"
            events.append(create_completion_event(
                completion=content,
                role="assistant",
                model=model,
            ))

    return events 