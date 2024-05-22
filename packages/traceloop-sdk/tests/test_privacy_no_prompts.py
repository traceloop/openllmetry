import os

import pytest
from openai import OpenAI
from opentelemetry.semconv.ai import SpanAttributes
from traceloop.sdk.decorators import workflow, task


@pytest.fixture(autouse=True)
def disable_trace_content():
    os.environ["TRACELOOP_TRACE_CONTENT"] = "false"
    yield
    os.environ["TRACELOOP_TRACE_CONTENT"] = "true"


@pytest.fixture
def openai_client():
    return OpenAI()


@pytest.mark.vcr
def test_simple_workflow(exporter, openai_client):
    @task(name="joke_creation")
    def create_joke():
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )
        return completion.choices[0].message.content

    @workflow(name="pirate_joke_generator")
    def joke_workflow():
        create_joke()

    joke_workflow()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
        "joke_creation.task",
        "pirate_joke_generator.workflow",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 15
    assert not open_ai_span.attributes.get("gen_ai.prompt.0.content")
    assert not open_ai_span.attributes.get("gen_ai.prompt.0.completions")
