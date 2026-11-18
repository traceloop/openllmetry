"""Non-ASCII prompt and completion text must stay readable in JSON span attributes."""

import json

import httpx
from anthropic import Anthropic
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

PROMPT = "Какая погода в Бостоне сегодня?"
SYSTEM = "Отвечай кратко"
COMPLETION = "В Бостоне сегодня ясно 🌤"
TOOL_DESCRIPTION = "Узнать погоду"

MESSAGE_RESPONSE = {
    "id": "msg_non_ascii",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5",
    "content": [{"type": "text", "text": COMPLETION}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 8},
}


def _client():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=MESSAGE_RESPONSE))
    return Anthropic(api_key="test", http_client=httpx.Client(transport=transport))


def test_message_attributes_keep_non_ascii_text(instrument_legacy, span_exporter):
    _client().messages.create(
        model="claude-sonnet-4-5",
        max_tokens=64,
        system=SYSTEM,
        messages=[{"role": "user", "content": PROMPT}],
        tools=[
            {
                "name": "get_weather",
                "description": TOOL_DESCRIPTION,
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
    )

    attributes = span_exporter.get_finished_spans()[0].attributes
    input_messages = attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    output_messages = attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    tool_definitions = attributes[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS]
    system_instructions = attributes[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS]

    assert PROMPT in input_messages
    assert SYSTEM in system_instructions
    assert COMPLETION in output_messages
    assert TOOL_DESCRIPTION in tool_definitions
    # The ASCII-escaped form of the prompt (what json.dumps produces by default).
    escaped_prompt = json.dumps(PROMPT)[1:-1]
    assert escaped_prompt not in input_messages
    # Still valid JSON carrying the same text.
    assert json.loads(output_messages)[0]["parts"][0]["content"] == COMPLETION


def test_json_dumps_escapes_only_unpaired_surrogates():
    from opentelemetry.instrumentation.anthropic.utils import json_dumps

    assert json_dumps({"text": "ясно"}) == '{"text": "ясно"}'
    lone_surrogate = "a" + chr(0xD800) + "b"
    dumped = json_dumps({"text": lone_surrogate})
    dumped.encode("utf-8")  # must not raise
    assert "ud800" in dumped
