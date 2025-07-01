
import json
import pytest

from openai import OpenAI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.mark.vcr
def test_responses(instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI):
    response = openai_client.responses.create(
        model="gpt-4.1-nano",
        input="What is the capital of France?",
    )
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "openai.response"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert (
        span.attributes["gen_ai.prompt.0.content"] == "What is the capital of France?"
    )
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert span.attributes["gen_ai.completion.0.content"] == response.output_text
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"


@pytest.mark.vcr
def test_responses_with_input_history(instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI):
    user_message = "Come up with an adjective in English. Respond with just one word."
    first_response = openai_client.responses.create(
        model="gpt-4.1-nano",
        input=user_message,
    )
    response = openai_client.responses.create(
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
                        "text": first_response.output_text,
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
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert (
        span.attributes["gen_ai.prompt.0.content"]
        == "Come up with an adjective in English. Respond with just one word."
    )
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert json.loads(span.attributes["gen_ai.prompt.1.content"]) == [
        {
            "type": "output_text",
            "text": first_response.output_text,
        }
    ]
    assert span.attributes["gen_ai.prompt.1.role"] == "assistant"
    assert (
        span.attributes["gen_ai.prompt.2.content"]
        == "Can you explain why you chose that word?"
    )
    assert span.attributes["gen_ai.prompt.2.role"] == "user"
    assert span.attributes["gen_ai.completion.0.content"] == response.output_text
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"


@pytest.mark.vcr
def test_responses_tool_calls(instrument_legacy, span_exporter: InMemorySpanExporter, openai_client: OpenAI):
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
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    ]
    openai_client.responses.create(
        model="gpt-4.1-nano",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": "What's the weather in London?"
            }
        ],
        tools=tools,
        tool_choice="auto"
    )

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    open_ai_span = spans[0]
    assert open_ai_span.name == "openai.response"
    assert open_ai_span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert open_ai_span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"

    assert open_ai_span.attributes["gen_ai.prompt.0.content"] == "What's the weather in London?"
    assert open_ai_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert open_ai_span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert open_ai_span.attributes["gen_ai.completion.0.tool_calls.0.name"] == "get_weather"
    assert open_ai_span.attributes["gen_ai.completion.0.tool_calls.0.arguments"] == '{"location":"London"}'
    assert open_ai_span.attributes["llm.request.functions.0.name"] == "get_weather"
    assert json.loads(open_ai_span.attributes["llm.request.functions.0.parameters"]) == {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            }
        },
        "required": ["location"]
    }
    assert open_ai_span.attributes["llm.request.functions.0.description"] == "Get the current weather for a location"

    assert (
        open_ai_span.attributes["gen_ai.completion.0.tool_calls.0.id"]
        == "fc_685ff89422ec819a977b2ea385bc9b6601c537ddeff5c2a2"
    )
    assert (
        open_ai_span.attributes["gen_ai.response.id"]
        == "resp_685ff8928dc4819aac45e085ba66838101c537ddeff5c2a2"
    )
