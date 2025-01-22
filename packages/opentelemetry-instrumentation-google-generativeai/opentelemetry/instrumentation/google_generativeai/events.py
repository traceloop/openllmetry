"""Event utility functions for Google Generative AI instrumentation."""

from typing import Any, Dict, List, Optional, Union

from opentelemetry.semconv_ai import (
    LLMRequestTypeValues,
    SpanAttributes,
)

def create_prompt_event(
    content: Union[str, Dict[str, Any]],
    role: Optional[str] = None,
    content_type: Optional[str] = None,
    model: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a prompt event for Google Generative AI.
    
    Args:
        content: The prompt content. Can be text or a dictionary for multi-modal content.
        role: The role of the prompt (e.g., "user", "system").
        content_type: The type of content (e.g., "text", "image").
        model: The model being used.
        prompt_tokens: Number of tokens in the prompt.
    """
    attributes = {
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.PROMPT,
        SpanAttributes.LLM_SYSTEM: "Google Generative AI",
    }

    # Handle multi-modal content
    if isinstance(content, dict):
        attributes[SpanAttributes.LLM_PROMPT_TEMPLATE] = str(content)
        if "text" in content:
            attributes[SpanAttributes.LLM_REQUEST_CONTENT_TYPE] = "text"
        elif "image" in content:
            attributes[SpanAttributes.LLM_REQUEST_CONTENT_TYPE] = "image"
    else:
        attributes[SpanAttributes.LLM_PROMPT_TEMPLATE] = content
        if content_type:
            attributes[SpanAttributes.LLM_REQUEST_CONTENT_TYPE] = content_type

    if role is not None:
        attributes[SpanAttributes.LLM_REQUEST_ROLE] = role

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
    safety_attributes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a completion event for Google Generative AI.
    
    Args:
        completion: The completion text.
        model: The model used for completion.
        completion_tokens: Number of tokens in the completion.
        role: The role of the completion (e.g., "assistant").
        content_type: The type of content.
        finish_reason: The reason for completion.
        safety_attributes: Safety ratings and attributes from the model.
    """
    attributes = {
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION,
        SpanAttributes.LLM_COMPLETION: completion,
        SpanAttributes.LLM_SYSTEM: "Google Generative AI",
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

    # Add safety attributes if provided
    if safety_attributes:
        for key, value in safety_attributes.items():
            attributes[f"llm.safety.{key}"] = str(value)

    return {
        "name": "completion",
        "attributes": attributes,
    }

def create_tool_call_event(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_output: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a tool call event for Google Generative AI.
    
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
        SpanAttributes.LLM_SYSTEM: "Google Generative AI",
    }

    if tool_output is not None:
        attributes[SpanAttributes.LLM_TOOL_OUTPUT] = str(tool_output)

    if model is not None:
        attributes[SpanAttributes.LLM_REQUEST_MODEL] = model

    return {
        "name": "tool_call",
        "attributes": attributes,
    }

def create_function_call_event(
    function_name: str,
    function_args: Dict[str, Any],
    function_output: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a function call event for Google Generative AI.
    
    Args:
        function_name: The name of the function being called.
        function_args: The arguments provided to the function.
        function_output: The output from the function.
        model: The model making the function call.
    """
    attributes = {
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.FUNCTION,
        "llm.function.name": function_name,
        "llm.function.args": str(function_args),
        SpanAttributes.LLM_SYSTEM: "Google Generative AI",
    }

    if function_output is not None:
        attributes["llm.function.output"] = str(function_output)

    if model is not None:
        attributes[SpanAttributes.LLM_REQUEST_MODEL] = model

    return {
        "name": "function_call",
        "attributes": attributes,
    } 