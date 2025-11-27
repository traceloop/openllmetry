import sys

import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="ibm-watson-ai requires python3.10"
)
@pytest.mark.vcr
def test_generate(exporter_legacy, watson_ai_model, log_exporter):
    if watson_ai_model is None:
        print("test_generate test skipped.")
        return
    watson_ai_model.generate(prompt="What is 1 + 1?")
    spans = exporter_legacy.get_finished_spans()
    watsonx_ai_span = spans[1]
    assert (
        watsonx_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.user"]
        == "What is 1 + 1?"
    )
    assert watsonx_ai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "Watsonx"
    assert watsonx_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    assert watsonx_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="ibm-watson-ai requires python3.10"
)
@pytest.mark.vcr
def test_generate_with_events_with_content(
    exporter_with_content, watson_ai_model, log_exporter
):
    if watson_ai_model is None:
        print("test_generate test skipped.")
        return
    response = watson_ai_model.generate(prompt="What is 1 + 1?")
    spans = exporter_with_content.get_finished_spans()
    watsonx_ai_span = spans[1]
    assert watsonx_ai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "Watsonx"
    assert watsonx_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": {"content": "What is 1 + 1?"}},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": response["stop_reason"],
        "message": {"content": response["generated_text"]},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="ibm-watson-ai requires python3.10"
)
@pytest.mark.vcr
def test_generate_with_with_events_no_content(
    exporter_with_no_content, watson_ai_model, log_exporter
):
    if watson_ai_model is None:
        print("test_generate test skipped.")
        return
    response = watson_ai_model.generate(prompt="What is 1 + 1?")
    spans = exporter_with_no_content.get_finished_spans()
    watsonx_ai_span = spans[1]
    assert watsonx_ai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "Watsonx"
    assert watsonx_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2
    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": response["stop_reason"],
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="ibm-watson-ai requires python3.10"
)
@pytest.mark.vcr
def test_generate_text_stream(exporter_legacy, watson_ai_model, log_exporter):
    if watson_ai_model is None:
        print("test_generate_text_stream test skipped.")
        return
    response = watson_ai_model.generate_text_stream(
        prompt="Write an epigram about the sun"
    )
    generated_text = ""
    for chunk in response:
        generated_text += chunk
    spans = exporter_legacy.get_finished_spans()
    watsonx_ai_span = spans[1]
    assert (
        watsonx_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.user"]
        == "Write an epigram about the sun"
    )
    assert watsonx_ai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "Watsonx"
    assert (
        watsonx_ai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == generated_text
    )
    assert watsonx_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="ibm-watson-ai requires python3.10"
)
@pytest.mark.vcr
def test_generate_text_stream_with_events_with_content(
    exporter_with_content, watson_ai_model, log_exporter
):
    if watson_ai_model is None:
        print("test_generate_text_stream test skipped.")
        return
    response = watson_ai_model.generate_text_stream(
        prompt="Write an epigram about the sun"
    )
    generated_text = ""
    for chunk in response:
        generated_text += chunk
    spans = exporter_with_content.get_finished_spans()
    watsonx_ai_span = spans[1]
    assert watsonx_ai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "Watsonx"
    assert watsonx_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": {"content": "Write an epigram about the sun"}},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": response["stop_reason"],
        "message": {"content": generated_text},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="ibm-watson-ai requires python3.10"
)
@pytest.mark.vcr
def test_generate_text_stream_with_events_with_no_content(
    exporter_with_no_content, watson_ai_model, log_exporter
):
    if watson_ai_model is None:
        print("test_generate_text_stream test skipped.")
        return
    response = watson_ai_model.generate_text_stream(
        prompt="Write an epigram about the sun"
    )
    generated_text = ""
    for chunk in response:
        generated_text += chunk
    spans = exporter_with_no_content.get_finished_spans()
    watsonx_ai_span = spans[1]
    assert watsonx_ai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "Watsonx"
    assert watsonx_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": response["stop_reason"],
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert (
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM)
        == GenAIAttributes.GenAiSystemValues.IBM_WATSONX_AI.value
    )

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
