import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_mistral_embeddings_legacy(
    instrument_legacy, mistralai_client, span_exporter, log_exporter
):
    mistralai_client.embeddings.create(
        model="mistral-embed",
        inputs="Tell me a joke about OpenTelemetry",
    )

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.embeddings"
    assert mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "MistralAI"
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "embedding"
    )
    assert not mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}")
        == "mistral-embed"
    )
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "cc93a06d10244e07a4b2604c20855c61"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_mistral_embeddings_with_events_with_content(
    instrument_with_content, mistralai_client, span_exporter, log_exporter
):
    response = mistralai_client.embeddings.create(
        model="mistral-embed",
        inputs="Tell me a joke about OpenTelemetry",
    )

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.embeddings"
    assert mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "MistralAI"
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "embedding"
    )
    assert not mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}")
        == "mistral-embed"
    )
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "e87386ab58e64d8c8c9f5b5175d7b1a9"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message = {"content": "Tell me a joke about OpenTelemetry"}
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {"content": response.data[0].embedding},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_mistral_embeddings_with_events_with_no_content(
    instrument_with_no_content, mistralai_client, span_exporter, log_exporter
):
    mistralai_client.embeddings.create(
        model="mistral-embed",
        inputs="Tell me a joke about OpenTelemetry",
    )

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.embeddings"
    assert mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "MistralAI"
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "embedding"
    )
    assert not mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}")
        == "mistral-embed"
    )
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "233321a98ca941b1a412c4c2a74e6a8d"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message = {}
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", user_message)

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_mistral_async_embeddings_legacy(
    instrument_legacy, mistralai_async_client, span_exporter, log_exporter
):
    await mistralai_async_client.embeddings.create_async(
        model="mistral-embed",
        inputs=["Tell me a joke about OpenTelemetry", "Tell me a joke about Traceloop"],
    )

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.embeddings"
    assert mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "MistralAI"
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "embedding"
    )
    assert not mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}")
        == "mistral-embed"
    )
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "d5656d6c70804024b2e0729a2f30ad55"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_mistral_async_embeddings_with_events_with_content(
    instrument_with_content, mistralai_async_client, span_exporter, log_exporter
):
    response = await mistralai_async_client.embeddings.create_async(
        model="mistral-embed",
        inputs=["Tell me a joke about OpenTelemetry", "Tell me a joke about Traceloop"],
    )

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.embeddings"
    assert mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "MistralAI"
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "embedding"
    )
    assert not mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}")
        == "mistral-embed"
    )
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "fd94466331664a4d8b2e3ff9bedd24f2"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    # Validate first user message Event
    user_message = {"content": "Tell me a joke about OpenTelemetry"}
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", user_message)

    # Validate second user message Event
    user_message = {"content": "Tell me a joke about Traceloop"}
    user_message_log = logs[1]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", user_message)

    # Validate the first ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {"content": response.data[0].embedding},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)

    # Validate the second ai response
    choice_event = {
        "index": 1,
        "finish_reason": "unknown",
        "message": {"content": response.data[1].embedding},
    }
    assert_message_in_logs(logs[3], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_mistral_async_embeddings_with_events_with_no_content(
    instrument_with_no_content, mistralai_async_client, span_exporter, log_exporter
):
    await mistralai_async_client.embeddings.create_async(
        model="mistral-embed",
        inputs=["Tell me a joke about OpenTelemetry", "Tell me a joke about Traceloop"],
    )

    spans = span_exporter.get_finished_spans()
    mistral_span = spans[0]
    assert mistral_span.name == "mistralai.embeddings"
    assert mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "MistralAI"
    assert (
        mistral_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "embedding"
    )
    assert not mistral_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        mistral_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}")
        == "mistral-embed"
    )
    assert (
        mistral_span.attributes.get("gen_ai.response.id")
        == "41c2e6bc6d4a463884a0729196c565b8"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    # Validate first user message Event
    user_message = {}
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", user_message)

    # Validate second user message Event
    user_message = {}
    user_message_log = logs[1]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", user_message)

    # Validate the first ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)

    # Validate the second ai response
    choice_event = {
        "index": 1,
        "finish_reason": "unknown",
        "message": {},
    }
    assert_message_in_logs(logs[3], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "mistral_ai"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
