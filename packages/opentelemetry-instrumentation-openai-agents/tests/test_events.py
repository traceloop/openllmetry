"""Test cases for event emission with use_legacy_attributes support."""

import pytest
from agents import Runner
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.semconv._incubating.attributes import (
    event_attributes as EventAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def assert_message_in_logs(log: ReadableLogRecord, event_name: str, expected_content: dict):
    """Helper function to assert event log structure."""
    log_record = log.log_record
    assert (
        log_record.attributes.get(EventAttributes.EVENT_NAME) == event_name
    ), f"Expected event name {event_name}, got {log_record.attributes.get(EventAttributes.EVENT_NAME)}"
    assert (
        log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM)
        == GenAIAttributes.GenAiSystemValues.OPENAI.value
    )

    if not expected_content:
        assert not log_record.body or dict(log_record.body) == {}
    else:
        assert log_record.body
        actual_body = dict(log_record.body)
        for key, value in expected_content.items():
            assert key in actual_body, f"Expected key {key} in body, got {actual_body}"
            assert (
                actual_body[key] == value
            ), f"Expected {key}={value}, got {actual_body[key]}"


@pytest.mark.vcr
def test_agent_with_legacy_attributes(exporter, test_agent):
    """Test that legacy attributes mode (default) uses span attributes, not events."""
    query = "What is AI?"
    Runner.run_sync(test_agent, query)

    spans = exporter.get_finished_spans()

    # Find the response span
    response_spans = [s for s in spans if s.name == "openai.response"]
    assert len(response_spans) >= 1
    response_span = response_spans[0]

    # Should have span attributes with prompts/completions
    assert response_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.role") == "user"
    assert (
        response_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content") == query
    )


@pytest.mark.skip(reason="Requires test isolation fixes for TracerProvider - to be addressed in follow-up")
@pytest.mark.vcr
def test_agent_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, test_agent
):
    """Test that events mode emits events with content when TRACELOOP_TRACE_CONTENT=True."""
    query = "What is AI?"
    Runner.run_sync(test_agent, query)

    spans = span_exporter.get_finished_spans()

    # Find the response span
    response_spans = [s for s in spans if s.name == "openai.response"]
    assert len(response_spans) >= 1
    response_span = response_spans[0]

    # Should NOT have prompt/completion span attributes (events are used instead)
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.content" not in response_span.attributes
    assert f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content" not in response_span.attributes

    # Should emit events
    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) >= 2
    ), f"Expected at least 2 events (message + choice), got {len(logs)}"

    # Validate user message Event
    user_message_log = None
    for log in logs:
        if (
            log.log_record.attributes.get(EventAttributes.EVENT_NAME)
            == "gen_ai.user.message"
        ):
            user_message_log = log
            break

    assert user_message_log is not None, "Expected gen_ai.user.message event"
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": query},
    )

    # Validate the choice event
    choice_log = None
    for log in logs:
        if log.log_record.attributes.get(EventAttributes.EVENT_NAME) == "gen_ai.choice":
            choice_log = log
            break

    assert choice_log is not None, "Expected gen_ai.choice event"
    # Just verify the event exists and has expected structure
    assert choice_log.log_record.body is not None


@pytest.mark.skip(reason="Requires test isolation fixes for TracerProvider - to be addressed in follow-up")
@pytest.mark.vcr
def test_agent_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, test_agent
):
    """Test that events mode emits events without content when TRACELOOP_TRACE_CONTENT=False."""
    query = "What is AI?"
    Runner.run_sync(test_agent, query)

    spans = span_exporter.get_finished_spans()

    # Find the response span
    response_spans = [s for s in spans if s.name == "openai.response"]
    assert len(response_spans) >= 1
    response_span = response_spans[0]

    # Should NOT have prompt/completion span attributes
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.content" not in response_span.attributes
    assert f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content" not in response_span.attributes

    # Should emit events
    logs = log_exporter.get_finished_logs()
    assert len(logs) >= 2, f"Expected at least 2 events, got {len(logs)}"

    # Validate user message Event (without content)
    user_message_log = None
    for log in logs:
        if (
            log.log_record.attributes.get(EventAttributes.EVENT_NAME)
            == "gen_ai.user.message"
        ):
            user_message_log = log
            break

    assert user_message_log is not None, "Expected gen_ai.user.message event"
    # Content should not be present when TRACELOOP_TRACE_CONTENT=False
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {},  # No content when TRACELOOP_TRACE_CONTENT=False
    )

    # Validate the choice event (without content)
    choice_log = None
    for log in logs:
        if log.log_record.attributes.get(EventAttributes.EVENT_NAME) == "gen_ai.choice":
            choice_log = log
            break

    assert choice_log is not None, "Expected gen_ai.choice event"


@pytest.mark.skip(reason="Requires test isolation fixes for TracerProvider - to be addressed in follow-up")
@pytest.mark.vcr
def test_agent_with_function_tool_events(
    instrument_with_content, span_exporter, log_exporter, function_tool_agent
):
    """Test that events are emitted correctly when agent uses function tools."""
    query = "What's the weather in London?"
    Runner.run_sync(function_tool_agent, query)

    spans = span_exporter.get_finished_spans()

    # Find the response span
    response_spans = [s for s in spans if s.name == "openai.response"]
    assert len(response_spans) >= 1

    # Should emit events
    logs = log_exporter.get_finished_logs()
    assert len(logs) >= 2, f"Expected at least 2 events, got {len(logs)}"

    # Should have user message event
    user_message_log = None
    for log in logs:
        if (
            log.log_record.attributes.get(EventAttributes.EVENT_NAME)
            == "gen_ai.user.message"
        ):
            user_message_log = log
            break

    assert user_message_log is not None
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": query},
    )

    # Should have choice event (may contain tool calls)
    choice_log = None
    for log in logs:
        if log.log_record.attributes.get(EventAttributes.EVENT_NAME) == "gen_ai.choice":
            choice_log = log
            break

    assert choice_log is not None
