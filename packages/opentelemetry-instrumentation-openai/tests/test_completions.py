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
        open_ai_span.attributes["llm.prompts.0.user"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get("llm.completions.0.content")


@pytest.mark.vcr
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
        open_ai_span.attributes["llm.prompts.0.user"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get("llm.completions.0.content")


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
        open_ai_span.attributes["llm.prompts.0.user"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get("llm.completions.0.content")


@pytest.mark.vcr
def test_completion_streaming(exporter, openai_client):
    response = openai_client.completions.create(
        model="davinci-002",
        prompt="Tell me a joke about opentelemetry",
        stream=True,
    )

    for part in response:
        pass

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes["llm.prompts.0.user"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get("llm.completions.0.content")


@pytest.mark.vcr
async def test_async_completion_streaming(exporter, async_openai_client):
    response = await async_openai_client.completions.create(
        model="davinci-002",
        prompt="Tell me a joke about opentelemetry",
        stream=True,
    )

    async for part in response:
        pass

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes["llm.prompts.0.user"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get("llm.completions.0.content")
