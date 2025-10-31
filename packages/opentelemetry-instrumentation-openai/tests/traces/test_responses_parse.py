"""Tests for OpenAI Responses.parse method with structured outputs."""

import json
import pytest
from openai import OpenAI, AsyncOpenAI
from opentelemetry.instrumentation.openai.utils import is_reasoning_supported
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import BaseModel
from typing import Optional


class Person(BaseModel):
    """Simple structured output for basic tests."""

    name: str
    age: int


class ContentCompliance(BaseModel):
    """Structured output for moderation tests."""

    is_violating: bool
    category: Optional[str]
    explanation_if_violating: Optional[str]


class WeatherInfo(BaseModel):
    """Structured output for tool call tests."""

    location: str
    temperature: int
    conditions: str


@pytest.mark.vcr
def test_responses_parse_basic(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    """Test basic responses.parse with simple structured output."""
    openai_client.responses.parse(
        model="gpt-4.1-nano",
        input="Jane, 54 years old",
        text_format=Person,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "openai.response"
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert span.attributes["gen_ai.prompt.0.content"] == "Jane, 54 years old"
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"

    # Verify the structured output was captured
    assert "gen_ai.completion.0.content" in span.attributes
    output_content = span.attributes["gen_ai.completion.0.content"]
    # The output should be JSON serialized
    parsed_output = json.loads(output_content)
    assert parsed_output["name"] == "Jane"
    assert parsed_output["age"] == 54

    # Verify token usage
    assert "gen_ai.usage.input_tokens" in span.attributes
    assert "gen_ai.usage.output_tokens" in span.attributes
    assert span.attributes["gen_ai.usage.input_tokens"] > 0
    assert span.attributes["gen_ai.usage.output_tokens"] > 0


@pytest.mark.vcr
def test_responses_parse_with_message_history(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    """Test responses.parse with message history."""
    openai_client.responses.parse(
        model="gpt-4.1-nano",
        input=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts person information.",
            },
            {
                "role": "user",
                "content": "Extract info: John Smith, 42 years old",
            },
        ],
        text_format=Person,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "openai.response"
    assert span.attributes["gen_ai.system"] == "openai"

    # Check system message
    assert (
        span.attributes["gen_ai.prompt.0.content"]
        == "You are a helpful assistant that extracts person information."
    )
    assert span.attributes["gen_ai.prompt.0.role"] == "system"

    # Check user message
    assert (
        span.attributes["gen_ai.prompt.1.content"]
        == "Extract info: John Smith, 42 years old"
    )
    assert span.attributes["gen_ai.prompt.1.role"] == "user"

    # Check response
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"
    output_content = span.attributes["gen_ai.completion.0.content"]
    parsed_output = json.loads(output_content)
    assert parsed_output["name"] == "John Smith"
    assert parsed_output["age"] == 42


@pytest.mark.vcr
def test_responses_parse_moderation(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    """Test responses.parse for content moderation use case."""
    openai_client.responses.parse(
        model="gpt-4.1-nano",
        input=[
            {
                "role": "system",
                "content": "Determine if the user input violates specific guidelines and explain if they do.",
            },
            {
                "role": "user",
                "content": "How do I prepare for a job interview?",
            },
        ],
        text_format=ContentCompliance,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "openai.response"
    assert span.attributes["gen_ai.system"] == "openai"

    # Verify structured output
    output_content = span.attributes["gen_ai.completion.0.content"]
    parsed_output = json.loads(output_content)
    assert parsed_output["is_violating"] is False
    assert parsed_output["category"] is None
    assert parsed_output["explanation_if_violating"] is None


@pytest.mark.vcr
def test_responses_parse_with_tools(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    """Test responses.parse with tool definitions."""
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        }
    ]

    openai_client.responses.parse(
        model="gpt-4.1-nano",
        input="What's the weather in Paris?",
        text_format=WeatherInfo,
        tools=tools,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "openai.response"
    assert span.attributes["gen_ai.system"] == "openai"

    # Verify tool was captured
    assert "llm.request.functions.0.name" in span.attributes
    assert span.attributes["llm.request.functions.0.name"] == "get_weather"
    assert (
        span.attributes["llm.request.functions.0.description"]
        == "Get the current weather for a location"
    )

    # When tools are used, the response may contain tool calls instead of content
    assert "gen_ai.completion.0.tool_calls.0.name" in span.attributes


@pytest.mark.vcr
@pytest.mark.skipif(
    not is_reasoning_supported(),
    reason="Reasoning is not supported in older OpenAI library versions",
)
def test_responses_parse_with_reasoning(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    """Test responses.parse with reasoning parameters."""
    openai_client.responses.parse(
        model="gpt-5-nano",
        input="Extract person info: Sarah, 28",
        text_format=Person,
        reasoning={"effort": "low", "summary": None},
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "openai.response"
    assert span.attributes["gen_ai.request.reasoning_effort"] == "low"
    assert span.attributes["gen_ai.request.reasoning_summary"] == ()
    assert span.attributes["gen_ai.response.reasoning_effort"] == "low"

    # Reasoning tokens should be tracked
    assert span.attributes["gen_ai.usage.reasoning_tokens"] >= 0

    # Verify structured output still works
    output_content = span.attributes["gen_ai.completion.0.content"]
    parsed_output = json.loads(output_content)
    assert parsed_output["name"] == "Sarah"
    assert parsed_output["age"] == 28


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_responses_parse_basic(
    instrument_legacy,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
):
    """Test async responses.parse with basic structured output."""
    await async_openai_client.responses.parse(
        model="gpt-4.1-nano",
        input="Mike, 35 years old",
        text_format=Person,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "openai.response"
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.prompt.0.content"] == "Mike, 35 years old"
    assert span.attributes["gen_ai.prompt.0.role"] == "user"

    # Verify structured output
    output_content = span.attributes["gen_ai.completion.0.content"]
    parsed_output = json.loads(output_content)
    assert parsed_output["name"] == "Mike"
    assert parsed_output["age"] == 35


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_responses_parse_with_message_history(
    instrument_legacy,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
):
    """Test async responses.parse with message history."""
    await async_openai_client.responses.parse(
        model="gpt-4.1-nano",
        input=[
            {
                "role": "system",
                "content": "Extract person data from text.",
            },
            {
                "role": "user",
                "content": "Parse this: Emily Jones, age 31",
            },
        ],
        text_format=Person,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "openai.response"
    assert (
        span.attributes["gen_ai.prompt.0.content"] == "Extract person data from text."
    )
    assert span.attributes["gen_ai.prompt.0.role"] == "system"
    assert (
        span.attributes["gen_ai.prompt.1.content"] == "Parse this: Emily Jones, age 31"
    )
    assert span.attributes["gen_ai.prompt.1.role"] == "user"

    # Verify structured output
    output_content = span.attributes["gen_ai.completion.0.content"]
    parsed_output = json.loads(output_content)
    assert parsed_output["name"] == "Emily Jones"
    assert parsed_output["age"] == 31


def test_responses_parse_exception(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    """Test that exceptions in responses.parse are properly traced."""
    openai_client.api_key = "invalid"

    with pytest.raises(Exception):
        openai_client.responses.parse(
            model="gpt-4.1-nano",
            input="Test input",
            text_format=Person,
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "openai.response"
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get("error.type") == "AuthenticationError"

    # Verify exception event
    events = span.events
    assert len(events) == 1
    event = events[0]
    assert event.name == "exception"
    assert event.attributes["exception.type"] == "openai.AuthenticationError"
    assert "Error code: 401" in event.attributes["exception.message"]


@pytest.mark.asyncio
async def test_async_responses_parse_exception(
    instrument_legacy,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
):
    """Test that exceptions in async responses.parse are properly traced."""
    async_openai_client.api_key = "invalid"

    with pytest.raises(Exception):
        await async_openai_client.responses.parse(
            model="gpt-4.1-nano",
            input="Test input",
            text_format=Person,
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "openai.response"
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get("error.type") == "AuthenticationError"

    # Verify exception event
    events = span.events
    assert len(events) == 1
    event = events[0]
    assert event.name == "exception"
    assert event.attributes["exception.type"] == "openai.AuthenticationError"
    assert "Error code: 401" in event.attributes["exception.message"]


@pytest.mark.vcr
def test_responses_parse_output_fallback(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    """Test that output is captured even if output_parsed is not available."""
    openai_client.responses.parse(
        model="gpt-4.1-nano",
        input="Alex, 29",
        text_format=Person,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    # Should have completion content even if parsing fails
    assert "gen_ai.completion.0.content" in span.attributes
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"

    # Verify response was successful
    assert span.status.status_code != StatusCode.ERROR


@pytest.mark.vcr
def test_responses_parse_with_instructions(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    """Test responses.parse with system instructions."""
    openai_client.responses.parse(
        model="gpt-4.1-nano",
        input="Robert, 45",
        instructions="You are an expert at extracting structured data.",
        text_format=Person,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "openai.response"

    # System instructions should be the first prompt
    assert (
        span.attributes["gen_ai.prompt.0.content"]
        == "You are an expert at extracting structured data."
    )
    assert span.attributes["gen_ai.prompt.0.role"] == "system"

    # User input should be next
    assert span.attributes["gen_ai.prompt.1.content"] == "Robert, 45"
    assert span.attributes["gen_ai.prompt.1.role"] == "user"

    # Verify structured output
    output_content = span.attributes["gen_ai.completion.0.content"]
    parsed_output = json.loads(output_content)
    assert parsed_output["name"] == "Robert"
    assert parsed_output["age"] == 45


@pytest.mark.vcr
def test_responses_parse_token_usage(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    """Test that token usage is properly tracked for responses.parse."""
    openai_client.responses.parse(
        model="gpt-4.1-nano",
        input="Lisa, 38 years old",
        text_format=Person,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    # Verify all token metrics are present
    assert "gen_ai.usage.input_tokens" in span.attributes
    assert "gen_ai.usage.output_tokens" in span.attributes
    assert "llm.usage.total_tokens" in span.attributes

    input_tokens = span.attributes["gen_ai.usage.input_tokens"]
    output_tokens = span.attributes["gen_ai.usage.output_tokens"]
    total_tokens = span.attributes["llm.usage.total_tokens"]

    assert input_tokens > 0
    assert output_tokens > 0
    assert total_tokens == input_tokens + output_tokens


@pytest.mark.vcr
def test_responses_parse_response_id(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    """Test that response ID is properly captured."""
    response = openai_client.responses.parse(
        model="gpt-4.1-nano",
        input="David, 52",
        text_format=Person,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    # Response ID should be present
    assert "gen_ai.response.id" in span.attributes
    response_id = span.attributes["gen_ai.response.id"]
    assert response_id.startswith("resp_")

    # Response ID should match the actual response
    assert response_id == response.id


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_responses_parse_with_tools(
    instrument_legacy,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
):
    """Test async responses.parse with tool definitions."""
    tools = [
        {
            "type": "function",
            "name": "search_database",
            "description": "Search for records in the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"],
            },
        }
    ]

    await async_openai_client.responses.parse(
        model="gpt-4.1-nano",
        input="Find person: Tom",
        text_format=Person,
        tools=tools,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "openai.response"

    # Verify tool was captured
    assert "llm.request.functions.0.name" in span.attributes
    assert span.attributes["llm.request.functions.0.name"] == "search_database"

    # When tools are used, the response may contain tool calls instead of content
    # Verify either content or tool calls are present
    assert (
        "gen_ai.completion.0.content" in span.attributes
        or "gen_ai.completion.0.tool_calls.0.name" in span.attributes
    )
