"""Unit tests for the events path (`use_attributes=False`) with message shapes the
recorded cassettes don't cover: tool calls and tool results."""

import copy
import json

import pytest
from opentelemetry.instrumentation.together.config import Config
from opentelemetry.instrumentation.together.event_emitter import (
    emit_completion_event,
    emit_prompt_events,
)
from opentelemetry.instrumentation.together.utils import TRACELOOP_TRACE_CONTENT
from opentelemetry.semconv_ai import LLMRequestTypeValues

CHAT = LLMRequestTypeValues.CHAT

TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})},
}

MESSAGES = [
    {"role": "user", "content": "What's the weather in Paris?"},
    {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
    {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
]

RESPONSE = {
    "id": "resp_1",
    "model": "openai/gpt-oss-120b",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
        }
    ],
}


class FakeEventLogger:
    def __init__(self):
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def bodies(self):
        return [(record.event_name, record.body) for record in self.records]


@pytest.fixture
def event_logger():
    return FakeEventLogger()


@pytest.fixture(autouse=True)
def events_enabled(monkeypatch):
    monkeypatch.setattr(Config, "use_legacy_attributes", False)


@pytest.fixture
def content_enabled(monkeypatch):
    monkeypatch.setenv(TRACELOOP_TRACE_CONTENT, "true")


@pytest.fixture
def content_disabled(monkeypatch):
    monkeypatch.setenv(TRACELOOP_TRACE_CONTENT, "false")


def test_prompt_events_include_tool_calls_and_tool_call_id(event_logger, content_enabled):
    emit_prompt_events(event_logger, CHAT, {"messages": MESSAGES})

    assert event_logger.bodies() == [
        ("gen_ai.user.message", {"content": "What's the weather in Paris?"}),
        ("gen_ai.assistant.message", {"content": None, "tool_calls": [TOOL_CALL]}),
        ("gen_ai.tool.message", {"content": "Sunny", "id": "call_1"}),
    ]


def test_choice_event_includes_tool_calls(event_logger, content_enabled):
    emit_completion_event(event_logger, CHAT, RESPONSE)

    assert event_logger.bodies() == [
        (
            "gen_ai.choice",
            {
                "index": 0,
                "finish_reason": "tool_call",
                "message": {"content": None},
                "tool_calls": [TOOL_CALL],
            },
        )
    ]


def test_tool_call_arguments_are_not_recorded_when_content_is_disabled(event_logger, content_disabled):
    messages = copy.deepcopy(MESSAGES)
    response = copy.deepcopy(RESPONSE)

    emit_prompt_events(event_logger, CHAT, {"messages": messages})
    emit_completion_event(event_logger, CHAT, response)

    tool_call_without_arguments = {"id": "call_1", "type": "function", "function": {"name": "get_weather"}}
    assert event_logger.bodies() == [
        ("gen_ai.user.message", {}),
        ("gen_ai.assistant.message", {"tool_calls": [tool_call_without_arguments]}),
        ("gen_ai.tool.message", {"id": "call_1"}),
        (
            "gen_ai.choice",
            {"index": 0, "finish_reason": "tool_call", "message": {}, "tool_calls": [tool_call_without_arguments]},
        ),
    ]
    # Stripping the arguments must not modify the app's own messages or response.
    assert messages == MESSAGES
    assert response == RESPONSE
