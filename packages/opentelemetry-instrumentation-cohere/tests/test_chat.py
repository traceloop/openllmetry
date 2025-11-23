import json
import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time in a given location",
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
    }
]


@pytest.mark.vcr
def test_cohere_chat_legacy(
    span_exporter, log_exporter, instrument_legacy, cohere_client
):
    res = cohere_client.chat(model="command", message="Tell me a joke, pirate style")

    spans = span_exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.chat"
    assert cohere_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert cohere_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == "command"
    assert (
        cohere_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
        == "Tell me a joke, pirate style"
    )
    assert (
        cohere_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == res.text
    )
    assert cohere_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 58
    assert cohere_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == cohere_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + cohere_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        cohere_span.attributes.get("gen_ai.response.id")
        == "440f51f4-3e47-44b6-a5d7-5ba33edcfc58"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_cohere_v2_chat_legacy(
    span_exporter, log_exporter, instrument_legacy, cohere_client_v2
):
    res = cohere_client_v2.chat(
        model="command", messages=[{"role": "user", "content": "Tell me a joke, pirate style"}]
    )

    spans = span_exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL) == "command"
    assert (
        cohere_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke, pirate style"
    )
    assert (
        json.loads(cohere_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content"))
        == [
            {
                "type": "text",
                "text": res.message.content[-1].text
            }
        ]
    )
    assert cohere_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 7
    assert cohere_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == cohere_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + cohere_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        cohere_span.attributes.get("gen_ai.response.id")
        == "83e3e297-264b-478e-9b22-5058386292ed"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_cohere_chat_legacy_with_streaming(
    span_exporter, log_exporter, instrument_legacy, cohere_client
):
    stream = cohere_client.chat_stream(model="command", message="Tell me a joke, pirate style")

    res = ""
    for chunk in stream:
        if chunk.event_type == "text-generation":
            res += chunk.text

    spans = span_exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL) == "command"
    assert (
        cohere_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke, pirate style"
    )
    assert (
        cohere_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == res
    )
    assert cohere_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 7
    assert cohere_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == cohere_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + cohere_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        cohere_span.attributes.get("gen_ai.response.id")
        == "12c00d26-e3bb-48c0-8c49-262155b57d64"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_cohere_v2_chat_legacy_with_streaming(
    span_exporter, log_exporter, instrument_legacy, cohere_client_v2
):
    stream = cohere_client_v2.chat_stream(
        model="command", messages=[{"role": "user", "content": "Tell me a joke, pirate style"}]
    )

    res = ""
    for chunk in stream:
        if chunk.type == "content-delta":
            res += chunk.delta.message.content.text

    spans = span_exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL) == "command"
    assert (
        cohere_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke, pirate style"
    )
    assert (
        json.loads(cohere_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content"))
        == [
            {
                "type": "text",
                "text": res,
                "thinking": None
            }
        ]
    )
    assert cohere_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 7
    assert cohere_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == cohere_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + cohere_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        cohere_span.attributes.get("gen_ai.response.id")
        == "6cd6ce61-bb3b-46f6-907e-fcfab45e51b6"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_cohere_v2_chat_legacy_with_tool_calls_and_history(
    span_exporter, log_exporter, instrument_legacy, cohere_client_v2
):
    user_prompt = "What is the weather and current time in Tokyo?"
    res1 = cohere_client_v2.chat(
        model="command-r",
        messages=[{"role": "user", "content": user_prompt}],
        tools=TOOLS,
    )
    res2 = cohere_client_v2.chat(
        model="command-r",
        messages=[
            {"role": "user", "content": user_prompt},
            {
                "role": "assistant",
                "content": res1.message.content,
                "tool_calls": res1.message.tool_calls
            },
            {
                "role": "tool",
                "tool_call_id": res1.message.tool_calls[0].id,
                "content": "4:20 PM"
            },
            {
                "role": "tool",
                "tool_call_id": res1.message.tool_calls[1].id,
                "content": "Sunny 20 degrees Celsius"
            },
        ],
        tools=TOOLS,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    sorted_spans = sorted(spans, key=lambda x: x.start_time)
    span1 = sorted_spans[0]
    span2 = sorted_spans[1]
    assert span1.name == "cohere.chat"
    assert span2.name == "cohere.chat"
    assert span1.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert span2.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert span1.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert span2.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert span1.attributes.get(SpanAttributes.LLM_REQUEST_MODEL) == "command-r"
    assert span2.attributes.get(SpanAttributes.LLM_REQUEST_MODEL) == "command-r"

    for span in [span1, span2]:
        assert (
            span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
            == user_prompt
        )
        assert span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.role") == "user"
        assert (
            span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name")
            == TOOLS[0].get("function").get("name")
        )
        assert (
            span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.description")
            == TOOLS[0].get("function").get("description")
        )
        assert (
            json.loads(span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.parameters"))
            == TOOLS[0].get("function").get("parameters")
        )
        assert (
            span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.name")
            == TOOLS[1].get("function").get("name")
        )
        assert (
            span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.description")
            == TOOLS[1].get("function").get("description")
        )
        assert (
            json.loads(span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.parameters"))
            == TOOLS[1].get("function").get("parameters")
        )
        assert (
            json.loads(span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.parameters"))
            == TOOLS[1].get("function").get("parameters")
        )

    assert (
        json.loads(span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content"))
        == [{
            "type": "text",
            "text": res1.message.tool_plan
        }]
    )
    assert (
        span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role")
        == "assistant"
    )
    assert (
        span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.id")
        == res1.message.tool_calls[0].id
    )
    assert (
        span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.name")
        == res1.message.tool_calls[0].function.name
    )
    assert (
        span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.arguments")
        == res1.message.tool_calls[0].function.arguments
    )
    assert (
        span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.id")
        == res1.message.tool_calls[1].id
    )
    assert (
        span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.name")
        == res1.message.tool_calls[1].function.name
    )
    assert (
        span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.arguments")
        == res1.message.tool_calls[1].function.arguments
    )

    assert (
        span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.name")
        == res1.message.tool_calls[1].function.name
    )
    assert (
        span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.arguments")
        == res1.message.tool_calls[1].function.arguments
    )

    assert (
        span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.id")
        == res1.message.tool_calls[0].id
    )
    assert (
        span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.name")
        == res1.message.tool_calls[0].function.name
    )
    assert (
        span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.arguments")
        == res1.message.tool_calls[0].function.arguments
    )
    assert (
        span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.id")
        == res1.message.tool_calls[1].id
    )
    assert (
        span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.name")
        == res1.message.tool_calls[1].function.name
    )
    assert (
        span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.arguments")
        == res1.message.tool_calls[1].function.arguments
    )

    assert span2.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.1.content") == res1.message.content
    assert span2.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.1.role") == "assistant"
    assert span2.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.2.content") == "4:20 PM"
    assert span2.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.2.role") == "tool"
    assert span2.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.3.content") == "Sunny 20 degrees Celsius"
    assert span2.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.3.role") == "tool"

    assert span2.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role") == "assistant"
    assert json.loads(span2.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")) == [{
        "type": "text",
        "text": res2.message.content[-1].text
    }]

    assert span1.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 62
    assert span1.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == span1.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + span1.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        span1.attributes.get("gen_ai.response.id")
        == "965405bb-b9da-4dc5-b329-e708a795e188"
    )
    assert span2.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 94
    assert span2.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == span2.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + span2.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        span2.attributes.get("gen_ai.response.id")
        == "00f7ec58-cd22-4b62-b068-0e7ece6bbf67"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_cohere_v2_chat_legacy_with_tool_calls_and_streaming(
    span_exporter, log_exporter, instrument_legacy, cohere_client_v2
):
    user_prompt = "What is the weather and current time in Tokyo?"
    res1 = cohere_client_v2.chat_stream(
        model="command-r",
        messages=[{"role": "user", "content": user_prompt}],
        tools=TOOLS,
    )

    plan = ""
    for chunk in res1:
        if chunk.type == "tool-plan-delta":
            plan += chunk.delta.message.tool_plan

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "cohere.chat"
    assert span.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL) == "command-r"

    assert (
        span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == user_prompt
    )
    assert span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.role") == "user"
    assert (
        span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name")
        == TOOLS[0].get("function").get("name")
    )
    assert (
        span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.description")
        == TOOLS[0].get("function").get("description")
    )
    assert (
        json.loads(span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.parameters"))
        == TOOLS[0].get("function").get("parameters")
    )
    assert (
        span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.name")
        == TOOLS[1].get("function").get("name")
    )
    assert (
        span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.description")
        == TOOLS[1].get("function").get("description")
    )
    assert (
        json.loads(span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.parameters"))
        == TOOLS[1].get("function").get("parameters")
    )

    assert (
        json.loads(span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content"))
        == [{
            "type": "text",
            "text": plan
        }]
    )
    assert span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role") == "assistant"
    assert (
        span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.id")
        == "get_time_dg5wwc00d8v5"
    )
    assert span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.name") == "get_time"
    assert (
        json.loads(span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.arguments"))
        == {"location": "Tokyo"}
    )
    assert (
        span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.id")
        == "get_weather_bc8241gkqss5"
    )
    assert (
        span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.name")
        == "get_weather"
    )
    assert (
        json.loads(span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.arguments"))
        == {"location": "Tokyo"}
    )

    assert span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 62
    assert span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        span.attributes.get("gen_ai.response.id")
        == "e6d757a9-f0f3-40fe-9b8a-44cdc3bd18a7"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_cohere_chat_with_events_with_content(
    span_exporter, log_exporter, instrument_with_content, cohere_client
):
    user_message = "Tell me a joke, pirate style"
    res = cohere_client.chat(model="command", message=user_message)

    spans = span_exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.chat"
    assert cohere_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert cohere_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == "command"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {"content": user_message})

    # Validate model response Event
    choice_event = {
        "index": 0,
        "finish_reason": "COMPLETE",
        "message": {"content": res.text},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_cohere_chat_with_events_with_no_content(
    span_exporter, log_exporter, instrument_with_no_content, cohere_client
):
    user_message = "Tell me a joke, pirate style"
    cohere_client.chat(model="command", message=user_message)

    spans = span_exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.chat"
    assert cohere_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert cohere_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == "command"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate model response Event
    choice_event = {
        "index": 0,
        "finish_reason": "COMPLETE",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_cohere_chat_legacy_async(
    span_exporter, log_exporter, instrument_legacy, async_cohere_client
):
    res = await async_cohere_client.chat(model="command", message="Tell me a joke, pirate style")

    spans = span_exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL) == "command"
    assert (
        cohere_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke, pirate style"
    )
    assert (
        cohere_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == res.text
    )
    assert cohere_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 7
    assert cohere_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == cohere_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + cohere_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        cohere_span.attributes.get("gen_ai.response.id")
        == "ea2d074c-4f25-47cb-bef8-b00dc2ae991b"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_cohere_chat_with_events_with_content_async(
    span_exporter, log_exporter, instrument_with_content, async_cohere_client
):
    user_message = "Tell me a joke, pirate style"
    res = await async_cohere_client.chat(model="command", message=user_message)

    spans = span_exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL) == "command"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {"content": user_message})

    # Validate model response Event
    choice_event = {
        "index": 0,
        "finish_reason": "COMPLETE",
        "message": {"content": res.text},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_cohere_chat_with_events_with_no_content_async(
    span_exporter, log_exporter, instrument_with_no_content, async_cohere_client
):
    user_message = "Tell me a joke, pirate style"
    await async_cohere_client.chat(model="command", message=user_message)

    spans = span_exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL) == "command"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate model response Event
    choice_event = {
        "index": 0,
        "finish_reason": "COMPLETE",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_cohere_chat_legacy_with_streaming_async(
    span_exporter, log_exporter, instrument_legacy, async_cohere_client
):
    stream = async_cohere_client.chat_stream(model="command", message="Tell me a joke, pirate style")

    res = ""
    async for chunk in stream:
        if chunk.event_type == "text-generation":
            res += chunk.text

    spans = span_exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL) == "command"
    assert (
        cohere_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke, pirate style"
    )
    assert (
        cohere_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == res
    )
    assert cohere_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 7
    assert cohere_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == cohere_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + cohere_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        cohere_span.attributes.get("gen_ai.response.id")
        == "dcdb9a85-a8dc-4f4c-9779-f7c1801248f3"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_cohere_v2_chat_legacy_with_streaming_async(
    span_exporter, log_exporter, instrument_legacy, async_cohere_client_v2
):
    stream = async_cohere_client_v2.chat_stream(
        model="command", messages=[{"role": "user", "content": "Tell me a joke, pirate style"}]
    )

    res = ""
    async for chunk in stream:
        if chunk.type == "content-delta":
            res += chunk.delta.message.content.text

    spans = span_exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL) == "command"
    assert (
        cohere_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke, pirate style"
    )
    assert (
        json.loads(cohere_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content"))
        == [
            {
                "type": "text",
                "text": res,
                "thinking": None
            }
        ]
    )
    assert cohere_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 7
    assert cohere_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == cohere_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + cohere_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        cohere_span.attributes.get("gen_ai.response.id")
        == "599ae0aa-0ef6-49e4-b7f4-e2fafc40ca2c"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_cohere_v2_chat_legacy_with_tool_calls_and_history_async(
    span_exporter, log_exporter, instrument_legacy, async_cohere_client_v2
):
    user_prompt = "What is the weather and current time in Tokyo?"
    res1 = await async_cohere_client_v2.chat(
        model="command-r7b-12-2024",
        messages=[{"role": "user", "content": user_prompt}],
        tools=TOOLS,
    )
    res2 = await async_cohere_client_v2.chat(
        model="command-r7b-12-2024",
        messages=[
            {"role": "user", "content": user_prompt},
            {
                "role": "assistant",
                "content": res1.message.content,
                "tool_calls": res1.message.tool_calls
            },
            {
                "role": "tool",
                "tool_call_id": res1.message.tool_calls[0].id,
                "content": "4:20 PM"
            },
            {
                "role": "tool",
                "tool_call_id": res1.message.tool_calls[1].id,
                "content": "Sunny 20 degrees Celsius"
            },
        ],
        tools=TOOLS,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    sorted_spans = sorted(spans, key=lambda x: x.start_time)
    span1 = sorted_spans[0]
    span2 = sorted_spans[1]
    assert span1.name == "cohere.chat"
    assert span2.name == "cohere.chat"
    assert span1.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert span2.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert span1.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert span2.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert span1.attributes.get(SpanAttributes.LLM_REQUEST_MODEL) == "command-r7b-12-2024"
    assert span2.attributes.get(SpanAttributes.LLM_REQUEST_MODEL) == "command-r7b-12-2024"

    for span in [span1, span2]:
        assert (
            span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
            == user_prompt
        )
        assert span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.role") == "user"
        assert (
            span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name")
            == TOOLS[0].get("function").get("name")
        )
        assert (
            span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.description")
            == TOOLS[0].get("function").get("description")
        )
        assert (
            json.loads(span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.parameters"))
            == TOOLS[0].get("function").get("parameters")
        )
        assert (
            span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.name")
            == TOOLS[1].get("function").get("name")
        )
        assert (
            span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.description")
            == TOOLS[1].get("function").get("description")
        )
        assert (
            json.loads(span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.parameters"))
            == TOOLS[1].get("function").get("parameters")
        )

    assert (
        json.loads(span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content"))
        == [{
            "type": "text",
            "text": res1.message.tool_plan
        }]
    )
    assert span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role") == "assistant"
    assert (
        span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.id")
        == res1.message.tool_calls[0].id
    )
    assert (
        span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.name")
        == res1.message.tool_calls[0].function.name
    )
    assert (
        span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.arguments")
        == res1.message.tool_calls[0].function.arguments
    )
    assert (
        span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.id")
        == res1.message.tool_calls[1].id
    )
    assert (
        span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.name")
        == res1.message.tool_calls[1].function.name
    )
    assert (
        span1.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.arguments")
        == res1.message.tool_calls[1].function.arguments
    )

    assert span2.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.1.content") == res1.message.content
    assert span2.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.1.role") == "assistant"
    assert span2.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.2.content") == "4:20 PM"
    assert span2.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.2.role") == "tool"
    assert span2.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.3.content") == "Sunny 20 degrees Celsius"
    assert span2.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.3.role") == "tool"

    assert span2.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role") == "assistant"
    assert json.loads(span2.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")) == [{
        "type": "text",
        "text": res2.message.content[-1].text
    }]

    assert span1.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 62
    assert span1.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == span1.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + span1.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        span1.attributes.get("gen_ai.response.id")
        == "b7391518-e53e-4486-98ea-fabffcde31c2"
    )
    assert span2.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 78
    assert span2.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == span2.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + span2.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        span2.attributes.get("gen_ai.response.id")
        == "8289d4ee-a83b-4ce2-b7e1-245d9778fcf5"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_cohere_v2_chat_legacy_with_tool_calls_and_streaming_async(
    span_exporter, log_exporter, instrument_legacy, async_cohere_client_v2
):
    user_prompt = "What is the weather and current time in Tokyo?"
    res1 = async_cohere_client_v2.chat_stream(
        model="command-r",
        messages=[{"role": "user", "content": user_prompt}],
        tools=TOOLS,
    )

    plan = ""
    async for chunk in res1:
        if chunk.type == "tool-plan-delta":
            plan += chunk.delta.message.tool_plan

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "cohere.chat"
    assert span.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL) == "command-r"

    assert (
        span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == user_prompt
    )
    assert span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.role") == "user"
    assert (
        span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name")
        == TOOLS[0].get("function").get("name")
    )
    assert (
        span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.description")
        == TOOLS[0].get("function").get("description")
    )
    assert (
        json.loads(span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.parameters"))
        == TOOLS[0].get("function").get("parameters")
    )
    assert (
        span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.name")
        == TOOLS[1].get("function").get("name")
    )
    assert (
        span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.description")
        == TOOLS[1].get("function").get("description")
    )
    assert (
        json.loads(span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.parameters"))
        == TOOLS[1].get("function").get("parameters")
    )

    assert (
        json.loads(span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content"))
        == [{
            "type": "text",
            "text": plan
        }]
    )
    assert span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role") == "assistant"
    assert (
        span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.id")
        == "get_time_mp1131yrhbga"
    )
    assert (
        span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.name")
        == "get_time"
    )
    assert (
        json.loads(span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.arguments"))
        == {"location": "Tokyo"}
    )
    assert (
        span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.id")
        == "get_weather_yxz1xx2m07cn"
    )
    assert span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.name") == "get_weather"
    assert (
        json.loads(span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.arguments"))
        == {"location": "Tokyo"}
    )

    assert span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 62
    assert span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    assert (
        span.attributes.get("gen_ai.response.id")
        == "ce257b09-c6da-4aed-b722-6384504180f5"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert (
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM)
        == GenAIAttributes.GenAiSystemValues.COHERE.value
    )

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
