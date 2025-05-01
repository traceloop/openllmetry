import pytest
from anthropic import AI_PROMPT, HUMAN_PROMPT
from opentelemetry.semconv_ai import SpanAttributes

from .utils import assert_message_in_logs, verify_metrics


@pytest.mark.vcr
def test_anthropic_completion_legacy(
    instrument_legacy, anthropic_client, span_exporter, reader, log_exporter
):
    anthropic_client.completions.create(
        prompt=f"{HUMAN_PROMPT}\nHello world\n{AI_PROMPT}",
        model="claude-instant-1.2",
        max_tokens_to_sample=2048,
        top_p=0.1,
    )
    try:
        anthropic_client.completions.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "anthropic.completion" for span in spans)

    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.user"]
        == f"{HUMAN_PROMPT}\nHello world\n{AI_PROMPT}"
    )
    assert anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "compl_01EjfrPvPEsRDRUKD6VoBxtK"
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics

    verify_metrics(
        resource_metrics, "claude-instant-1.2", ignore_zero_input_tokens=True
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_anthropic_completion_with_events_with_content(
    instrument_with_content, anthropic_client, span_exporter, reader, log_exporter
):
    prompt = f"{HUMAN_PROMPT}\nHello world\n{AI_PROMPT}"
    anthropic_client.completions.create(
        prompt=prompt,
        model="claude-instant-1.2",
        max_tokens_to_sample=2048,
        top_p=0.1,
    )
    try:
        anthropic_client.completions.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "anthropic.completion" for span in spans)

    anthropic_span = spans[0]
    assert anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.user"] == prompt
    assert anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "compl_01EjfrPvPEsRDRUKD6VoBxtK"
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics

    verify_metrics(
        resource_metrics, "claude-instant-1.2", ignore_zero_input_tokens=True
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message = {"content": prompt}
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop_sequence",
        "message": {"content": " Hello!"},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_anthropic_completion_with_events_with_no_content(
    instrument_with_no_content, anthropic_client, span_exporter, reader, log_exporter
):
    anthropic_client.completions.create(
        prompt=f"{HUMAN_PROMPT}\nHello world\n{AI_PROMPT}",
        model="claude-instant-1.2",
        max_tokens_to_sample=2048,
        top_p=0.1,
    )
    try:
        anthropic_client.completions.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "anthropic.completion" for span in spans)

    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.user"]
        == f"{HUMAN_PROMPT}\nHello world\n{AI_PROMPT}"
    )
    assert anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "compl_01EjfrPvPEsRDRUKD6VoBxtK"
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics

    verify_metrics(
        resource_metrics, "claude-instant-1.2", ignore_zero_input_tokens=True
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop_sequence",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)
