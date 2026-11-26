import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from .helpers import (
    MODEL,
    PROMPT,
    assert_common_span_attributes,
    assert_logs_in_span_context,
    assert_message_in_logs,
    assert_metrics,
    assert_no_content_attributes,
    get_single_span,
    input_messages,
    otel_finish_reason,
    output_messages,
)

TEXT_COMPLETION = GenAIAttributes.GenAiOperationNameValues.TEXT_COMPLETION.value


@pytest.mark.vcr
def test_together_completion_legacy(
    instrument_legacy, caller, span_exporter, log_exporter, reader
):
    response = caller.completion(
        model=MODEL,
        prompt=PROMPT,
    )

    together_span = get_single_span(span_exporter, TEXT_COMPLETION)
    assert_common_span_attributes(together_span, TEXT_COMPLETION, response)
    assert input_messages(together_span) == [
        {"role": "user", "parts": [{"type": "text", "content": PROMPT}]}
    ]
    choice = response.choices[0]
    assert output_messages(together_span) == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": choice.text}],
            "finish_reason": otel_finish_reason(choice),
        }
    ]
    assert_metrics(reader, TEXT_COMPLETION, response)

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_together_completion_with_events_with_content(
    instrument_with_content, caller, span_exporter, log_exporter, reader
):
    response = caller.completion(
        model=MODEL,
        prompt=PROMPT,
    )

    together_span = get_single_span(span_exporter, TEXT_COMPLETION)
    assert_common_span_attributes(together_span, TEXT_COMPLETION, response)
    assert_no_content_attributes(together_span)
    assert_metrics(reader, TEXT_COMPLETION, response)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2
    assert_logs_in_span_context(logs, together_span)

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {"content": PROMPT})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": otel_finish_reason(response.choices[0]),
        "message": {"content": response.choices[0].text},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_together_completion_with_events_with_no_content(
    instrument_with_no_content, caller, span_exporter, log_exporter, reader
):
    response = caller.completion(
        model=MODEL,
        prompt=PROMPT,
    )

    together_span = get_single_span(span_exporter, TEXT_COMPLETION)
    assert_common_span_attributes(together_span, TEXT_COMPLETION, response)
    assert_no_content_attributes(together_span)
    assert_metrics(reader, TEXT_COMPLETION, response)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2
    assert_logs_in_span_context(logs, together_span)

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": otel_finish_reason(response.choices[0]),
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)
