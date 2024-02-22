import pytest
import json
from openai import OpenAI
from traceloop.sdk.decorators import workflow, task


@pytest.fixture
def openai_client():
    return OpenAI()


@pytest.mark.vcr
def test_simple_workflow(exporter, openai_client):
    @task(name="something_creator")
    def create_something(what: str, subject: str):
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Tell me a {what} about {subject}"}],
        )
        return completion.choices[0].message.content

    @workflow(name="pirate_joke_generator")
    def joke_workflow():
        return create_something("joke", subject="OpenTelemetry")

    joke = joke_workflow()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
        "something_creator.task",
        "pirate_joke_generator.workflow",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes["llm.prompts.0.content"]
        == "Tell me a joke about OpenTelemetry"
    )
    assert open_ai_span.attributes.get("llm.completions.0.content")

    task_span = spans[1]
    assert json.loads(task_span.attributes["traceloop.entity.input"]) == {
        "args": ["joke"],
        "kwargs": {"subject": "OpenTelemetry"},
    }

    assert json.loads(task_span.attributes.get("traceloop.entity.output")) == joke


def test_unserializable_workflow(exporter):
    @task(name="unserializable_task")
    def unserializable_task(obj: object):
        return object()

    @workflow(name="unserializable_workflow")
    def unserializable_workflow(obj: object):
        return unserializable_task(obj)

    unserializable_task(object())

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["unserializable_task.task"]
