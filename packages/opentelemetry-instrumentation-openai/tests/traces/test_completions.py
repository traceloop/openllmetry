import os

import pytest


@pytest.mark.vcr
def test_completion(exporter, openai_client):
    openai_client.completions.create(
        model="davinci-002",
        prompt="Tell me a joke about opentelemetry",
    )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes["gen_ai.prompt.0.user"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get("gen_ai.completion.0.content")
    assert (
        open_ai_span.attributes.get("openai.api_base") == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get("llm.is_streaming") is False


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_completion(exporter, async_openai_client):
    await async_openai_client.completions.create(
        model="davinci-002",
        prompt="Tell me a joke about opentelemetry",
    )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes["gen_ai.prompt.0.user"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get("gen_ai.completion.0.content")


@pytest.mark.vcr
def test_completion_langchain_style(exporter, openai_client):
    openai_client.completions.create(
        model="davinci-002",
        prompt=["Tell me a joke about opentelemetry"],
    )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes["gen_ai.prompt.0.user"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get("gen_ai.completion.0.content")


@pytest.mark.vcr
def test_completion_streaming(exporter, openai_client):
    # set os env for token usage record in stream mode
    original_value = os.environ.get("TRACELOOP_STREAM_TOKEN_USAGE")
    os.environ["TRACELOOP_STREAM_TOKEN_USAGE"] = "true"

    try:
        response = openai_client.completions.create(
            model="davinci-002",
            prompt="Tell me a joke about opentelemetry",
            stream=True,
        )

        for _ in response:
            pass

        spans = exporter.get_finished_spans()
        assert [span.name for span in spans] == [
            "openai.completion",
        ]
        open_ai_span = spans[0]
        assert (
            open_ai_span.attributes["gen_ai.prompt.0.user"]
            == "Tell me a joke about opentelemetry"
        )
        assert open_ai_span.attributes.get("gen_ai.completion.0.content")
        assert (
            open_ai_span.attributes.get("openai.api_base") == "https://api.openai.com/v1/"
        )

        # check token usage attributes for stream
        completion_tokens = open_ai_span.attributes.get("gen_ai.usage.completion_tokens")
        prompt_tokens = open_ai_span.attributes.get("gen_ai.usage.prompt_tokens")
        total_tokens = open_ai_span.attributes.get("llm.usage.total_tokens")
        assert completion_tokens and prompt_tokens and total_tokens
        assert completion_tokens + prompt_tokens == total_tokens
    finally:
        # unset env
        if original_value is None:
            del os.environ["TRACELOOP_STREAM_TOKEN_USAGE"]
        else:
            os.environ["TRACELOOP_STREAM_TOKEN_USAGE"] = original_value


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_completion_streaming(exporter, async_openai_client):
    response = await async_openai_client.completions.create(
        model="davinci-002",
        prompt="Tell me a joke about opentelemetry",
        stream=True,
    )

    async for _ in response:
        pass

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes["gen_ai.prompt.0.user"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get("gen_ai.completion.0.content")
    assert (
        open_ai_span.attributes.get("openai.api_base") == "https://api.openai.com/v1/"
    )
