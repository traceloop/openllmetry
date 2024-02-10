import pytest
from openai import OpenAI
from traceloop.sdk.decorators import workflow, task


@pytest.fixture
def openai_client():
    return OpenAI()


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
    assert (
        open_ai_span.attributes["llm.prompts.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get("llm.completions.0.content")
