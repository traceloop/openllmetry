import json

import pytest
from opentelemetry.instrumentation.openai.utils import is_reasoning_supported
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

PROMPT_FILTER_KEY = "prompt_filter_results"
PROMPT_ERROR = "prompt_error"


@pytest.mark.vcr
def test_chat(instrument_legacy, span_exporter, log_exporter, azure_openai_client):
    azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://traceloop-stg.openai.azure.com/openai/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9HpbZPf84KZFiQG6fdY0KVtIwHyIa"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_chat_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, azure_openai_client
):
    response = azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://traceloop-stg.openai.azure.com/openai/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9HpbZPf84KZFiQG6fdY0KVtIwHyIa"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {"content": "Tell me a joke about opentelemetry"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "content": response.choices[0].message.content,
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_chat_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, azure_openai_client
):
    azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://traceloop-stg.openai.azure.com/openai/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9HpbZPf84KZFiQG6fdY0KVtIwHyIa"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "stop", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_chat_content_filtering(
    instrument_legacy, span_exporter, log_exporter, azure_openai_client
):
    azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert (
        open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
        == "FILTERED"
    )
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://traceloop-stg.openai.azure.com/openai/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9HpyGSWv1hoKdGaUaiFhfxzTEVlZo"
    )

    content_filter_json = open_ai_span.attributes.get(
        f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content_filter_results"
    )

    assert len(content_filter_json) > 0

    content_filter_results = json.loads(content_filter_json)

    assert content_filter_results["hate"]["filtered"] is True
    assert content_filter_results["hate"]["severity"] == "high"
    assert content_filter_results["self_harm"]["filtered"] is False
    assert content_filter_results["self_harm"]["severity"] == "safe"

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_chat_content_filtering_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, azure_openai_client
):
    azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://traceloop-stg.openai.azure.com/openai/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9HpyGSWv1hoKdGaUaiFhfxzTEVlZo"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {"content": "Tell me a joke about opentelemetry"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "content_filter",
        "message": {"content": None, "role": "unknown"},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_chat_content_filtering_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, azure_openai_client
):
    azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://traceloop-stg.openai.azure.com/openai/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9HpyGSWv1hoKdGaUaiFhfxzTEVlZo"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "content_filter",
        "message": {"role": "unknown"},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_prompt_content_filtering(
    instrument_legacy, span_exporter, log_exporter, azure_openai_client
):
    azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    assert isinstance(
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.{PROMPT_ERROR}"], str
    )

    error = json.loads(
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.{PROMPT_ERROR}"]
    )

    assert "innererror" in error

    assert "content_filter_result" in error["innererror"]

    assert error["innererror"]["code"] == "ResponsibleAIPolicyViolation"

    assert error["innererror"]["content_filter_result"]["hate"]["filtered"]

    assert error["innererror"]["content_filter_result"]["hate"]["severity"] == "high"

    assert error["innererror"]["content_filter_result"]["sexual"]["filtered"] is False

    assert error["innererror"]["content_filter_result"]["sexual"]["severity"] == "safe"

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_prompt_content_filtering_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, azure_openai_client
):
    azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    assert isinstance(
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.{PROMPT_ERROR}"], str
    )

    error = json.loads(
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.{PROMPT_ERROR}"]
    )

    assert "innererror" in error

    assert "content_filter_result" in error["innererror"]

    assert error["innererror"]["code"] == "ResponsibleAIPolicyViolation"

    assert error["innererror"]["content_filter_result"]["hate"]["filtered"]

    assert error["innererror"]["content_filter_result"]["hate"]["severity"] == "high"

    assert error["innererror"]["content_filter_result"]["sexual"]["filtered"] is False

    assert error["innererror"]["content_filter_result"]["sexual"]["severity"] == "safe"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 1, "Should not have a response event because of the error."

    # Validate user message Event
    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {"content": "Tell me a joke about opentelemetry"},
    )


@pytest.mark.vcr
def test_prompt_content_filtering_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, azure_openai_client
):
    azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    assert isinstance(
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.{PROMPT_ERROR}"], str
    )

    error = json.loads(
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.{PROMPT_ERROR}"]
    )

    assert "innererror" in error

    assert "content_filter_result" in error["innererror"]

    assert error["innererror"]["code"] == "ResponsibleAIPolicyViolation"

    assert error["innererror"]["content_filter_result"]["hate"]["filtered"]

    assert error["innererror"]["content_filter_result"]["hate"]["severity"] == "high"

    assert error["innererror"]["content_filter_result"]["sexual"]["filtered"] is False

    assert error["innererror"]["content_filter_result"]["sexual"]["severity"] == "safe"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 1, "Should not have a response event because of the error."

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})


@pytest.mark.vcr
def test_chat_streaming(
    instrument_legacy, span_exporter, log_exporter, azure_openai_client
):
    response = azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    chunk_count = 0
    for _ in response:
        chunk_count += 1

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://traceloop-stg.openai.azure.com/openai/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    events = open_ai_span.events
    assert len(events) == chunk_count

    # prompt filter results
    prompt_filter_results = json.loads(
        open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.{PROMPT_FILTER_KEY}")
    )
    assert prompt_filter_results[0]["prompt_index"] == 0
    assert (
        prompt_filter_results[0]["content_filter_results"]["hate"]["severity"] == "safe"
    )
    assert (
        prompt_filter_results[0]["content_filter_results"]["self_harm"]["filtered"]
        is False
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9HpbaAXyt0cAnlWvI8kUAFpZt5jyQ"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_chat_streaming_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, azure_openai_client
):
    response = azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    chunk_count = 0
    for _ in response:
        chunk_count += 1

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://traceloop-stg.openai.azure.com/openai/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    events = open_ai_span.events
    assert len(events) == chunk_count

    # prompt filter results
    prompt_filter_results = json.loads(
        open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.{PROMPT_FILTER_KEY}")
    )
    assert prompt_filter_results[0]["prompt_index"] == 0
    assert (
        prompt_filter_results[0]["content_filter_results"]["hate"]["severity"] == "safe"
    )
    assert (
        prompt_filter_results[0]["content_filter_results"]["self_harm"]["filtered"]
        is False
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9HpbaAXyt0cAnlWvI8kUAFpZt5jyQ"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {"content": "Tell me a joke about opentelemetry"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "content": (
                "Why did the Opentelemetry developer bring a ladder to the coding "
                "competition? \n\nBecause they wanted to reach new traces!"
            ),
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_chat_streaming_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, azure_openai_client
):
    response = azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    chunk_count = 0
    for _ in response:
        chunk_count += 1

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://traceloop-stg.openai.azure.com/openai/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    events = open_ai_span.events
    assert len(events) == chunk_count

    # prompt filter results
    prompt_filter_results = json.loads(
        open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.{PROMPT_FILTER_KEY}")
    )
    assert prompt_filter_results[0]["prompt_index"] == 0
    assert (
        prompt_filter_results[0]["content_filter_results"]["hate"]["severity"] == "safe"
    )
    assert (
        prompt_filter_results[0]["content_filter_results"]["self_harm"]["filtered"]
        is False
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9HpbaAXyt0cAnlWvI8kUAFpZt5jyQ"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "stop", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_chat_async_streaming(
    instrument_legacy, span_exporter, log_exporter, async_azure_openai_client
):
    response = await async_azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    chunk_count = 0
    async for _ in response:
        chunk_count += 1

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://traceloop-stg.openai.azure.com/openai/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    # Only assert token usage if API provides it (Existing cassetes of Azure OpenAI may not include usage in streaming)
    completion_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    prompt_tokens = open_ai_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    total_tokens = open_ai_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)
    if completion_tokens and prompt_tokens and total_tokens:
        assert completion_tokens + prompt_tokens == total_tokens

    events = open_ai_span.events
    assert len(events) == chunk_count
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9HpbbsSaH8U6amSDAwdA2WzMeDdLB"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_chat_async_streaming_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, async_azure_openai_client
):
    response = await async_azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    chunk_count = 0
    async for _ in response:
        chunk_count += 1

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://traceloop-stg.openai.azure.com/openai/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    # Only assert token usage if API provides it (Existing cassetes of Azure OpenAI may not include usage in streaming)
    completion_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    prompt_tokens = open_ai_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    total_tokens = open_ai_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)
    if completion_tokens and prompt_tokens and total_tokens:
        assert completion_tokens + prompt_tokens == total_tokens

    events = open_ai_span.events
    assert len(events) == chunk_count
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9HpbbsSaH8U6amSDAwdA2WzMeDdLB"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {"content": "Tell me a joke about opentelemetry"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "content": (
                "Why did the OpenTelemetry project become a stand-up comedian?\n\n"
                "Because it wanted to trace its steps and observe its laughter-per-minute rate!"
            ),
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_chat_async_streaming_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, async_azure_openai_client
):
    response = await async_azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    chunk_count = 0
    async for _ in response:
        chunk_count += 1

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://traceloop-stg.openai.azure.com/openai/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    # Only assert token usage if API provides it (Existing cassetes of Azure OpenAI may not include usage in streaming)
    completion_tokens = open_ai_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    prompt_tokens = open_ai_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
    total_tokens = open_ai_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)
    if completion_tokens and prompt_tokens and total_tokens:
        assert completion_tokens + prompt_tokens == total_tokens

    events = open_ai_span.events
    assert len(events) == chunk_count
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9HpbbsSaH8U6amSDAwdA2WzMeDdLB"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "stop", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
@pytest.mark.skipif(not is_reasoning_supported(),
                    reason="Reasoning is not supported in older OpenAI library versions")
def test_chat_reasoning(instrument_legacy, span_exporter,
                        log_exporter, azure_openai_client):
    azure_openai_client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {
                "role": "user",
                "content": "Count r's in strawberry"
            }
        ],
        reasoning_effort="low",
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[-1]

    assert span.attributes["gen_ai.request.reasoning_effort"] == "low"
    assert span.attributes["gen_ai.usage.reasoning_tokens"] > 0


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert (
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM)
        == GenAIAttributes.GenAiSystemValues.OPENAI.value
    )

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
