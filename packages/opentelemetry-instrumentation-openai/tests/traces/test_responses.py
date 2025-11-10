import pytest

from openai import OpenAI
from opentelemetry.instrumentation.openai.utils import is_reasoning_supported
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.mark.vcr
def test_responses(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    _ = openai_client.responses.create(
        model="gpt-4.1-nano",
        input="What is the capital of France?",
    )
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "openai.response"
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    # assert (
    #     span.attributes["gen_ai.prompt.0.content"] == "What is the capital of France?"
    # )
    # assert span.attributes["gen_ai.prompt.0.role"] == "user"


@pytest.mark.vcr
def test_responses_with_input_history(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    user_message = "Come up with an adjective in English. Respond with just one word."
    first_response = openai_client.responses.create(
        model="gpt-4.1-nano",
        input=user_message,
    )
    _ = openai_client.responses.create(
        model="gpt-4.1-nano",
        input=[
            {
                "role": "user",
                "content": user_message,
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": first_response.output[0].content[0].text,
                    }
                ],
            },
            {"role": "user", "content": "Can you explain why you chose that word?"},
        ],
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    span = spans[1]
    assert span.name == "openai.response"
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    # assert (
    #     span.attributes["gen_ai.prompt.0.content"]
    #     == "Come up with an adjective in English. Respond with just one word."
    # )
    # assert span.attributes["gen_ai.prompt.0.role"] == "user"
    # assert json.loads(span.attributes["gen_ai.prompt.1.content"]) == [
    #     {
    #         "type": "output_text",
    #         "text": first_response.output[0].content[0].text,
    #     }
    # ]
    # assert span.attributes["gen_ai.prompt.1.role"] == "assistant"
    # assert (
    #     span.attributes["gen_ai.prompt.2.content"]
    #     == "Can you explain why you chose that word?"
    # )
    # assert span.attributes["gen_ai.prompt.2.role"] == "user"


@pytest.mark.vcr
def test_responses_tool_calls(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
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
    openai_client.responses.create(
        model="gpt-4.1-nano",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": "What's the weather in London?",
            }
        ],
        tools=tools,
        tool_choice="auto",
    )

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    span = spans[0]
    assert span.name == "openai.response"
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"

    # assert span.attributes["gen_ai.prompt.0.content"] == "What's the weather in London?"
    # assert span.attributes["gen_ai.prompt.0.role"] == "user"
    # assert span.attributes["llm.request.functions.0.name"] == "get_weather"
    # assert json.loads(span.attributes["llm.request.functions.0.parameters"]) == {
    #     "type": "object",
    #     "properties": {
    #         "location": {
    #             "type": "string",
    #             "description": "The city and state, e.g. San Francisco, CA"
    #         }
    #     },
    #     "required": ["location"]
    # }
    # assert span.attributes["llm.request.functions.0.description"] == "Get the current weather for a location"

    # assert (
    #     span.attributes["gen_ai.completion.0.tool_calls.0.id"]
    #     == "fc_685ff89422ec819a977b2ea385bc9b6601c537ddeff5c2a2"
    # )
    assert (
        span.attributes["gen_ai.response.id"]
        == "resp_685ff8928dc4819aac45e085ba66838101c537ddeff5c2a2"
    )


@pytest.mark.vcr
@pytest.mark.skipif(
    not is_reasoning_supported(),
    reason="Reasoning is not supported in older OpenAI library versions",
)
def test_responses_reasoning(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    openai_client.responses.create(
        model="gpt-5-nano",
        input="Count r's in strawberry",
        reasoning={"effort": "low", "summary": None},
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    # assert span.attributes["gen_ai.request.reasoning_effort"] == "low"
    # assert span.attributes["gen_ai.request.reasoning_summary"] == ()

    # assert span.attributes["gen_ai.response.reasoning_effort"] == "low"
    # When reasoning summary is None/empty, the attribute should not be set
    assert "gen_ai.completion.0.reasoning" not in span.attributes

    # assert span.attributes["gen_ai.usage.reasoning_tokens"] > 0


@pytest.mark.vcr
@pytest.mark.skipif(
    not is_reasoning_supported(),
    reason="Reasoning is not supported in older OpenAI library versions",
)
def test_responses_reasoning_dict_issue(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    """Test for issue #3350 - reasoning dict causing invalid type warning"""
    openai_client.responses.create(
        model="gpt-5-nano",
        input="Explain why the sky is blue",
        reasoning={"effort": "medium", "summary": "auto"},
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    # Verify the reasoning attributes are properly set without causing warnings
    # assert span.attributes["gen_ai.request.reasoning_effort"] == "medium"
    # assert span.attributes["gen_ai.request.reasoning_summary"] == "auto"
    # This should not cause an "Invalid type dict" warning and should contain serialized reasoning
    assert "gen_ai.completion.0.reasoning" not in span.attributes
    # The reasoning should be serialized as JSON since it contains complex data
    # reasoning_attr = span.attributes["gen_ai.completion.0.reasoning"]
    # assert isinstance(reasoning_attr, str)
    # Should be valid JSON containing reasoning summary data
    # import json
    # parsed_reasoning = json.loads(reasoning_attr)
    # assert isinstance(parsed_reasoning, (dict, list))  # Could be dict or list depending on response structure


@pytest.mark.vcr
def test_responses_streaming(
    instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    """Test for streaming responses.create() - reproduces customer issue"""
    stream = openai_client.responses.create(
        model="gpt-4.1-nano",
        input="Tell me a three sentence bedtime story about a unicorn.",
        stream=True,
    )

    # Consume the stream
    full_text = ""
    for item in stream:
        if hasattr(item, "type") and item.type == "response.output_text.delta":
            if hasattr(item, "delta") and item.delta:
                full_text += item.delta
        elif hasattr(item, "delta") and item.delta:
            if hasattr(item.delta, "text") and item.delta.text:
                full_text += item.delta.text

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span but got {len(spans)}"

    span = spans[0]
    assert span.name == "openai.response"
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert full_text != "", "Should have received streaming content"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_responses_streaming_async(
    instrument_legacy, span_exporter: InMemorySpanExporter, async_openai_client
):
    """Test for async streaming responses.create() - reproduces customer issue"""
    stream = await async_openai_client.responses.create(
        model="gpt-4.1-nano",
        input="Tell me a three sentence bedtime story about a unicorn.",
        stream=True,
    )

    full_text = ""
    async for item in stream:
        if hasattr(item, "type") and item.type == "response.output_text.delta":
            if hasattr(item, "delta") and item.delta:
                full_text += item.delta
        elif hasattr(item, "delta") and item.delta:
            if hasattr(item.delta, "text") and item.delta.text:
                full_text += item.delta.text

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span but got {len(spans)}"

    span = spans[0]
    assert span.name == "openai.response"
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert full_text != "", "Should have received streaming content"
