"""Tests for emit_prompt_events (issue #2513).

A list of Content objects passed to generate_content is a conversation
history: each Content must be emitted as its own message event with its own
role, not stringified into a single user message.
"""
from unittest.mock import patch

from opentelemetry.instrumentation.vertexai.event_emitter import emit_prompt_events
from vertexai.generative_models import Content, Part


class FakeEventLogger:
    def __init__(self):
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _emit(args):
    logger = FakeEventLogger()
    with patch(
        "opentelemetry.instrumentation.vertexai.event_emitter.should_emit_events",
        return_value=True,
    ):
        emit_prompt_events(args, logger)
    return logger.records


class TestEmitPromptEvents:
    def test_content_history_emits_one_event_per_message(self):
        args = [
            [
                Content(role="user", parts=[Part.from_text("What's 2+2?")]),
                Content(role="model", parts=[Part.from_text("5")]),
                Content(role="user", parts=[Part.from_text("really?")]),
            ]
        ]

        records = _emit(args)

        assert [r.event_name for r in records] == [
            "gen_ai.user.message",
            "gen_ai.assistant.message",
            "gen_ai.user.message",
        ]
        assert [r.body["content"] for r in records] == [
            "What's 2+2?",
            "5",
            "really?",
        ]

    def test_string_argument_emits_single_user_event(self):
        records = _emit(("Hello, world!",))

        assert len(records) == 1
        assert records[0].event_name == "gen_ai.user.message"
        assert records[0].body["content"] == "Hello, world!"

    def test_parts_list_emits_single_user_event(self):
        records = _emit(([Part.from_text("hello"), "world"],))

        assert len(records) == 1
        assert records[0].event_name == "gen_ai.user.message"
        assert records[0].body["content"] == "hello\nworld"
