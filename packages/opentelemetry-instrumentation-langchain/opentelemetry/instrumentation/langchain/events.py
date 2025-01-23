"""Event utility functions for Langchain instrumentation."""

from typing import Any, Dict, List, Optional, Union
from langchain_core.messages import BaseMessage

from opentelemetry.semconv_ai import (
    LLMRequestTypeValues,
    SpanAttributes,
)

def _message_type_to_role(message_type: str) -> str:
    """Convert Langchain message type to standard role."""
    if message_type == "human":
        return "user"
    elif message_type == "system":
        return "system"
    elif message_type == "ai":
        return "assistant"
    else:
        return "unknown"

def create_prompt_event(
    content: Union[str, Dict[str, Any], BaseMessage],
    role: Optional[str] = None,
    content_type: Optional[str] = None,
    model: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a prompt event for Langchain.
    
    Args:
        content: The prompt content. Can be text, message, or dictionary.
        role: The role of the prompt (e.g., "user", "system").
        content_type: The type of content (e.g., "text").
        model: The model being used.
        prompt_tokens: Number of tokens in the prompt.
    """
    attributes = {
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.PROMPT,
        SpanAttributes.LLM_SYSTEM: "Langchain",
    }

    if isinstance(content, BaseMessage):
        attributes[SpanAttributes.LLM_PROMPT_TEMPLATE] = content.content
        role = _message_type_to_role(content.type)
    elif isinstance(content, dict):
        attributes[SpanAttributes.LLM_PROMPT_TEMPLATE] = str(content)
    else:
        attributes[SpanAttributes.LLM_PROMPT_TEMPLATE] = str(content)

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
    """Create a completion event for Langchain.
    
    Args:
        completion: The completion text.
        model: The model used for completion.
        completion_tokens: Number of tokens in the completion.
        role: The role of the completion (e.g., "assistant").
        content_type: The type of content.
        finish_reason: The reason for completion.
    """
    attributes = {
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION,
        SpanAttributes.LLM_COMPLETION: completion,
        SpanAttributes.LLM_SYSTEM: "Langchain",
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

def create_chain_event(
    chain_type: str,
    inputs: Dict[str, Any],
    outputs: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a chain event for Langchain.
    
    Args:
        chain_type: The type of chain being executed.
        inputs: The inputs to the chain.
        outputs: The outputs from the chain.
        model: The model being used.
    """
    attributes = {
        SpanAttributes.LLM_REQUEST_TYPE: "chain",
        SpanAttributes.LLM_SYSTEM: "Langchain",
        "llm.chain.type": chain_type,
        "llm.chain.input": str(inputs),
    }

    if outputs is not None:
        attributes["llm.chain.output"] = str(outputs)

    if model is not None:
        attributes[SpanAttributes.LLM_REQUEST_MODEL] = model

    return {
        "name": "chain",
        "attributes": attributes,
    }

def create_tool_event(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_output: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a tool event for Langchain.
    
    Args:
        tool_name: The name of the tool being called.
        tool_input: The input provided to the tool.
        tool_output: The output from the tool.
        model: The model making the tool call.
    """
    attributes = {
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.TOOL,
        SpanAttributes.LLM_TOOL_NAME: tool_name,
        SpanAttributes.LLM_TOOL_INPUT: str(tool_input),
        SpanAttributes.LLM_SYSTEM: "Langchain",
    }

    if tool_output is not None:
        attributes[SpanAttributes.LLM_TOOL_OUTPUT] = str(tool_output)

    if model is not None:
        attributes[SpanAttributes.LLM_REQUEST_MODEL] = model

    return {
        "name": "tool_call",
        "attributes": attributes,
    } 