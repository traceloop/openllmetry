import sys

import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires python3.9")
@pytest.mark.vcr
def test_replicate_llama_stream_legacy(
    instrument_legacy, replicate_client, span_exporter, log_exporter
):
    model_version = "meta/llama-2-70b-chat"
    for event in replicate_client.stream(
        model_version,
        input={
            "prompt": "tell me a joke about opentelemetry",
        },
    ):
        continue

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "replicate.stream",
    ]

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires python3.9")
@pytest.mark.vcr
def test_replicate_llama_stream_with_events_with_content(
    instrument_with_content, replicate_client, span_exporter, log_exporter
):
    model_version = "meta/llama-2-70b-chat"
    response = ""
    for event in replicate_client.stream(
        model_version,
        input={
            "prompt": "tell me a joke about opentelemetry",
        },
    ):
        response += str(event)

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "replicate.stream",
    ]

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "tell me a joke about opentelemetry"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {"content": response},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires python3.9")
@pytest.mark.vcr
def test_replicate_llama_stream_with_events_with_no_content(
    instrument_with_no_content, replicate_client, span_exporter, log_exporter
):
    model_version = "meta/llama-2-70b-chat"
    for event in replicate_client.stream(
        model_version,
        input={
            "prompt": "tell me a joke about opentelemetry",
        },
    ):
        continue

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "replicate.stream",
    ]

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "replicate"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
