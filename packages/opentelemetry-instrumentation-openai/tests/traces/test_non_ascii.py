"""Verify that non-ASCII text is preserved (not \\uXXXX escaped) in the
gen_ai.input.messages, gen_ai.output.messages and gen_ai.tool.definitions
span attributes produced by the OpenAI instrumentation.

This mirrors the fix applied to the LangChain instrumentation for the same
issue (see PR #3696): json.dumps(...) calls that build these attributes must
pass ensure_ascii=False so multi-byte UTF-8 text round-trips unescaped.
"""

import asyncio
from unittest.mock import MagicMock

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from opentelemetry.instrumentation.openai.shared import _set_tool_definitions_json
from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
    _set_input_messages,
    _set_output_messages,
)
from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
    RealtimeEventProcessor,
    RealtimeSessionState,
    RealtimeSessionWrapper,
)

CYRILLIC_TEXT = "Привет, как дела?"


def _get_attribute(mock_span, name):
    for call in mock_span.set_attribute.call_args_list:
        if call.args[0] == name:
            return call.args[1]
    return None


def test_set_input_messages_preserves_cyrillic():
    span = MagicMock()
    span.is_recording.return_value = True

    _set_input_messages(span, [{"role": "user", "content": CYRILLIC_TEXT}])

    raw = _get_attribute(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES)
    assert raw is not None
    assert CYRILLIC_TEXT in raw
    assert "\\u" not in raw


def test_set_output_messages_preserves_cyrillic():
    span = MagicMock()
    span.is_recording.return_value = True

    choices = [
        {
            "message": {"role": "assistant", "content": CYRILLIC_TEXT},
            "finish_reason": "stop",
        }
    ]
    _set_output_messages(span, choices)

    raw = _get_attribute(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
    assert raw is not None
    assert CYRILLIC_TEXT in raw
    assert "\\u" not in raw


def test_set_tool_definitions_preserves_cyrillic():
    span = MagicMock()
    span.is_recording.return_value = True

    tool_defs = [
        {
            "type": "function",
            "name": "get_weather",
            "description": CYRILLIC_TEXT,
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    _set_tool_definitions_json(span, tool_defs)

    raw = _get_attribute(span, GenAIAttributes.GEN_AI_TOOL_DEFINITIONS)
    assert raw is not None
    assert CYRILLIC_TEXT in raw
    assert "\\u" not in raw


def test_handle_session_updated_preserves_cyrillic_modalities():
    state = RealtimeSessionState(tracer=MagicMock(), model="gpt-4o-realtime")
    state.session_span = MagicMock()
    state.session_span.is_recording.return_value = True
    processor = RealtimeEventProcessor(state)

    class _Session:
        modalities = [CYRILLIC_TEXT]
        temperature = None

        def model_dump(self):
            return {"modalities": self.modalities, "temperature": self.temperature}

    event = MagicMock()
    event.session = _Session()

    processor.handle_session_updated(event)

    raw = _get_attribute(state.session_span, GenAIAttributes.GEN_AI_OUTPUT_TYPE)
    assert raw is not None
    assert CYRILLIC_TEXT in raw
    assert "\\u" not in raw


def test_session_wrapper_update_preserves_cyrillic_instructions():
    state = RealtimeSessionState(tracer=MagicMock(), model="gpt-4o-realtime")
    state.session_span = MagicMock()
    state.session_span.is_recording.return_value = True

    inner_session = MagicMock()

    async def _update(**kwargs):
        return MagicMock()

    inner_session.update = _update
    wrapper = RealtimeSessionWrapper(inner_session, state)

    asyncio.run(wrapper.update(session={"instructions": CYRILLIC_TEXT}))

    raw = _get_attribute(state.session_span, GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS)
    assert raw is not None
    assert CYRILLIC_TEXT in raw
    assert "\\u" not in raw
