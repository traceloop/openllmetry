"""Test event-based functionality for LlamaIndex instrumentation."""

import pytest
from unittest.mock import Mock, patch

from opentelemetry._events import Event, EventLogger
from opentelemetry.instrumentation.llamaindex.events import (
    create_prompt_event,
    create_completion_event,
    create_tool_call_event,
)
from opentelemetry.instrumentation.llamaindex.config import Config
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes


def test_create_prompt_event():
    """Test creating a prompt event."""
    prompt = {"prompt": "What is the capital of France?"}
    event = create_prompt_event(prompt)

    assert isinstance(event, Event)
    assert event.name == "gen_ai.prompt"
    assert event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.LLAMAINDEX.value
    assert event.body == prompt


def test_create_completion_event():
    """Test creating a completion event."""
    completion = {"completion": "The capital of France is Paris."}
    event = create_completion_event(completion)

    assert isinstance(event, Event)
    assert event.name == "gen_ai.completion"
    assert event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.LLAMAINDEX.value
    assert event.body == completion


def test_create_tool_call_event():
    """Test creating a tool call event."""
    tool_call = {
        "name": "search",
        "arguments": {"query": "capital of France"}
    }
    event = create_tool_call_event(tool_call)

    assert isinstance(event, Event)
    assert event.name == "gen_ai.tool_call"
    assert event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == GenAIAttributes.GenAiSystemValues.LLAMAINDEX.value
    assert event.body == {"tool_call": tool_call}


def test_event_without_content():
    """Test events with capture_content=False."""
    prompt = {"prompt": "What is the capital of France?"}
    event = create_prompt_event(prompt, capture_content=False)

    assert isinstance(event, Event)
    assert event.name == "gen_ai.prompt"
    assert event.body == {}


def test_event_with_trace_context():
    """Test events with trace context."""
    prompt = {"prompt": "What is the capital of France?"}
    trace_id = 123
    span_id = 456
    trace_flags = 1

    event = create_prompt_event(
        prompt,
        trace_id=trace_id,
        span_id=span_id,
        trace_flags=trace_flags
    )

    assert event.trace_id == trace_id
    assert event.span_id == span_id
    assert event.trace_flags == trace_flags 