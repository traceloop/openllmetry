"""Tool results are distinct prompt events, not copies of adjacent messages."""

import pytest
from mistralai.models import ToolMessage, UserMessage
from opentelemetry.instrumentation.mistralai import _emit_message_events


@pytest.mark.parametrize("tool_first", [True, False])
def test_tool_message_emits_its_own_content(tool_first, instrument_with_content, logger_provider, log_exporter):
    user = UserMessage(content="what is 2+2?")
    tool = ToolMessage(content="4", tool_call_id="call_1")
    messages = [tool, user] if tool_first else [user, tool]

    event_logger = logger_provider.get_logger(__name__)
    _emit_message_events("mistralai.chat", (), {"messages": messages}, event_logger)

    logs = log_exporter.get_finished_logs()
    assert [(log.log_record.event_name, dict(log.log_record.body)["content"]) for log in logs] == (
        [("gen_ai.tool.message", "4"), ("gen_ai.user.message", "what is 2+2?")]
        if tool_first else
        [("gen_ai.user.message", "what is 2+2?"), ("gen_ai.tool.message", "4")]
    )


class _FutureMessage:
    role = "assistant"
    content = "future content"


def test_unknown_message_type_does_not_repeat_previous(instrument_with_content, logger_provider, log_exporter):
    _emit_message_events(
        "mistralai.chat", (),
        {"messages": [UserMessage(content="old prompt"), _FutureMessage()]},
        logger_provider.get_logger(__name__),
    )
    logs = log_exporter.get_finished_logs()
    assert [(log.log_record.event_name, dict(log.log_record.body)["content"]) for log in logs] == [
        ("gen_ai.user.message", "old prompt"),
        ("gen_ai.assistant.message", "future content"),
    ]
