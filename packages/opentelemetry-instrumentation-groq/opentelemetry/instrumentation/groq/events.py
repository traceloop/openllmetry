"""Event utilities for Groq instrumentation."""

import json
from typing import Any, Dict, Optional

from opentelemetry.semconv_ai import EventAttributes, EventNames


def create_prompt_event(
    content: str,
    role: str,
    model: Optional[str] = None,
    content_type: str = "text",
    prompt_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a prompt event with the given attributes.

    Args:
        content: The content of the prompt.
        role: The role of the prompt (e.g., "user", "system", "assistant").
        model: The model used for the prompt.
        content_type: The type of content (e.g., "text", "image").
        prompt_tokens: The number of tokens in the prompt.

    Returns:
        A dictionary containing the event name and attributes.
    """
    attributes = {
        EventAttributes.LLM_PROMPTS_CONTENT: content,
        EventAttributes.LLM_PROMPTS_ROLE: role,
        EventAttributes.LLM_PROMPTS_CONTENT_TYPE: content_type,
    }

    if model:
        attributes[EventAttributes.LLM_MODEL] = model

    if prompt_tokens is not None:
        attributes[EventAttributes.LLM_TOKEN_COUNT] = prompt_tokens

    return {
        "name": EventNames.LLM_PROMPTS,
        "attributes": attributes,
    }


def create_completion_event(
    completion: str,
    model: Optional[str] = None,
    role: Optional[str] = None,
    finish_reason: Optional[str] = None,
    content_filter_results: Optional[Dict[str, Any]] = None,
    completion_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a completion event with the given attributes.

    Args:
        completion: The completion text.
        model: The model used for the completion.
        role: The role of the completion (e.g., "assistant").
        finish_reason: The reason the completion finished.
        content_filter_results: Results from content filtering.
        completion_tokens: The number of tokens in the completion.

    Returns:
        A dictionary containing the event name and attributes.
    """
    attributes = {
        EventAttributes.LLM_COMPLETIONS_CONTENT: completion,
    }

    if model:
        attributes[EventAttributes.LLM_MODEL] = model

    if role:
        attributes[EventAttributes.LLM_COMPLETIONS_ROLE] = role

    if finish_reason:
        attributes[EventAttributes.LLM_COMPLETIONS_FINISH_REASON] = finish_reason

    if content_filter_results:
        attributes[EventAttributes.LLM_COMPLETIONS_CONTENT_FILTER_RESULTS] = json.dumps(content_filter_results)

    if completion_tokens is not None:
        attributes[EventAttributes.LLM_TOKEN_COUNT] = completion_tokens

    return {
        "name": EventNames.LLM_COMPLETIONS,
        "attributes": attributes,
    }


def create_tool_call_event(
    tool_name: str,
    tool_input: Dict[str, Any],
    model: Optional[str] = None,
    tool_output: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a tool call event with the given attributes.

    Args:
        tool_name: The name of the tool being called.
        tool_input: The input parameters for the tool call.
        model: The model making the tool call.
        tool_output: The output from the tool call.

    Returns:
        A dictionary containing the event name and attributes.
    """
    attributes = {
        EventAttributes.LLM_TOOL_NAME: tool_name,
        EventAttributes.LLM_TOOL_INPUT: json.dumps(tool_input),
    }

    if model:
        attributes[EventAttributes.LLM_MODEL] = model

    if tool_output:
        attributes[EventAttributes.LLM_TOOL_OUTPUT] = json.dumps(tool_output)

    return {
        "name": EventNames.LLM_TOOL_CALLS,
        "attributes": attributes,
    }


def create_function_call_event(
    function_name: str,
    function_args: Dict[str, Any],
    model: Optional[str] = None,
    function_output: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a function call event with the given attributes.

    Args:
        function_name: The name of the function being called.
        function_args: The arguments passed to the function.
        model: The model making the function call.
        function_output: The output from the function call.

    Returns:
        A dictionary containing the event name and attributes.
    """
    attributes = {
        EventAttributes.LLM_FUNCTION_NAME: function_name,
        EventAttributes.LLM_FUNCTION_ARGS: json.dumps(function_args),
    }

    if model:
        attributes[EventAttributes.LLM_MODEL] = model

    if function_output:
        attributes[EventAttributes.LLM_FUNCTION_OUTPUT] = json.dumps(function_output)

    return {
        "name": EventNames.LLM_FUNCTION_CALLS,
        "attributes": attributes,
    } 
 