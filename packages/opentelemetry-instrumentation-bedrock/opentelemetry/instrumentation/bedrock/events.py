"""Event utility functions for Bedrock instrumentation."""

from typing import Any, Dict, List, Optional

from opentelemetry.semconv_ai import (
    LLMRequestTypeValues,
    SpanAttributes,
)

def create_prompt_event(
    prompt: str,
    prompt_tokens: Optional[int] = None,
    role: Optional[str] = None,
    content_type: Optional[str] = None,
    vendor: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a prompt event for Bedrock."""
    attributes = {
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.PROMPT,
        SpanAttributes.LLM_PROMPT_TEMPLATE: prompt,
    }

    if prompt_tokens is not None:
        attributes[SpanAttributes.LLM_TOKEN_COUNT] = prompt_tokens

    if role is not None:
        attributes[SpanAttributes.LLM_REQUEST_ROLE] = role

    if content_type is not None:
        attributes[SpanAttributes.LLM_REQUEST_CONTENT_TYPE] = content_type

    if vendor is not None:
        attributes[SpanAttributes.LLM_SYSTEM] = vendor

    if model is not None:
        attributes[SpanAttributes.LLM_REQUEST_MODEL] = model

    return {
        "name": "prompt",
        "attributes": attributes,
    }

def create_completion_event(
    completion: str,
    completion_tokens: Optional[int] = None,
    role: Optional[str] = None,
    content_type: Optional[str] = None,
    vendor: Optional[str] = None,
    model: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a completion event for Bedrock."""
    attributes = {
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION,
        SpanAttributes.LLM_COMPLETION: completion,
    }

    if completion_tokens is not None:
        attributes[SpanAttributes.LLM_TOKEN_COUNT] = completion_tokens

    if role is not None:
        attributes[SpanAttributes.LLM_RESPONSE_ROLE] = role

    if content_type is not None:
        attributes[SpanAttributes.LLM_RESPONSE_CONTENT_TYPE] = content_type

    if vendor is not None:
        attributes[SpanAttributes.LLM_SYSTEM] = vendor

    if model is not None:
        attributes[SpanAttributes.LLM_RESPONSE_MODEL] = model

    if finish_reason is not None:
        attributes[SpanAttributes.LLM_RESPONSE_FINISH_REASON] = finish_reason

    return {
        "name": "completion",
        "attributes": attributes,
    }

def create_tool_call_event(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_output: Optional[Dict[str, Any]] = None,
    vendor: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a tool call event for Bedrock."""
    attributes = {
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.TOOL,
        SpanAttributes.LLM_TOOL_NAME: tool_name,
        SpanAttributes.LLM_TOOL_INPUT: str(tool_input),
    }

    if tool_output is not None:
        attributes[SpanAttributes.LLM_TOOL_OUTPUT] = str(tool_output)

    if vendor is not None:
        attributes[SpanAttributes.LLM_SYSTEM] = vendor

    if model is not None:
        attributes[SpanAttributes.LLM_REQUEST_MODEL] = model

    return {
        "name": "tool_call",
        "attributes": attributes,
    } 