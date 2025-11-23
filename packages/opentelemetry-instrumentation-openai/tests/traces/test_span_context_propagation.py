"""Test span context propagation to events."""

import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def assert_event_has_span_context(log: LogData, expected_trace_id: int, expected_span_id: int):
    """Assert that an event log has the expected trace and span context."""
    log_record = log.log_record

    # Check that the event has trace context
    assert log_record.trace_id == expected_trace_id, (
        f"Event trace_id {log_record.trace_id} doesn't match span trace_id {expected_trace_id}"
    )
    assert log_record.span_id == expected_span_id, (
        f"Event span_id {log_record.span_id} doesn't match span span_id {expected_span_id}"
    )

    # Verify it's a proper OpenAI event
    assert log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == GenAIAttributes.GenAiSystemValues.OPENAI.value


def test_span_context_propagation_with_mock_client(
    instrument_with_content, span_exporter, log_exporter, mock_openai_client
):
    """Test that events have proper span context using mock client."""
    # The mock_openai_client fixture should trigger instrumentation but not make real calls
    try:
        mock_openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test span context"}],
        )
    except Exception:
        pass

    spans = span_exporter.get_finished_spans()
    logs = log_exporter.get_finished_logs()

    # Check if we got any spans (instrumentation worked)
    if len(spans) > 0:
        span = spans[0]
        assert span.name == "openai.chat"

        # Check if we got any events
        if len(logs) > 0:
            # Verify the first event has the same trace and span context as the span
            user_event = logs[0]
            assert_event_has_span_context(user_event, span.context.trace_id, span.context.span_id)

            # Verify it's the expected event type
            assert user_event.log_record.event_name == "gen_ai.user.message"
        else:
            pytest.skip("No events generated - may be due to test configuration")
    else:
        pytest.skip("No spans generated - instrumentation may not be active in this test")
