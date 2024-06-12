import json

import pytest
from openai import OpenAI
from opentelemetry.semconv.ai import SpanAttributes
from traceloop.sdk.decorators import workflow, task, aworkflow, atask


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
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about OpenTelemetry"
    )
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")

    task_span = spans[1]
    assert json.loads(task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "args": ["joke"],
        "kwargs": {"subject": "OpenTelemetry"},
    }

    assert (
        json.loads(task_span.attributes.get(SpanAttributes.TRACELOOP_ENTITY_OUTPUT))
        == joke
    )


@pytest.mark.vcr
def test_streaming_workflow(exporter, openai_client):

    @workflow(name="pirate_joke_generator")
    def joke_workflow():
        response_stream = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Tell me a joke about OpenTelemetry"}
            ],
            stream=True,
        )
        for chunk in response_stream:
            yield chunk

    joke_stream = joke_workflow()
    for _ in joke_stream:
        pass

    spans = exporter.get_finished_spans()
    assert set([span.name for span in spans]) == set(
        [
            "openai.chat",
            "pirate_joke_generator.workflow",
        ]
    )
    workflow_span = next(
        span for span in spans if span.name == "pirate_joke_generator.workflow"
    )
    openai_span = next(span for span in spans if span.name == "openai.chat")

    assert openai_span.parent.span_id == workflow_span.context.span_id
    assert openai_span.end_time <= workflow_span.end_time


def test_unrelated_entities(exporter):
    @workflow(name="workflow_1")
    def workflow_1():
        return

    @task(name="task_1")
    def task_1():
        return

    workflow_1()
    task_1()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["workflow_1.workflow", "task_1.task"]

    workflow_1_span = spans[0]
    task_1_span = spans[1]

    assert (
        workflow_1_span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "workflow_1"
    )
    assert workflow_1_span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "workflow"

    assert task_1_span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "task_1"
    assert task_1_span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "task"
    assert task_1_span.parent is None


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


@pytest.mark.asyncio
async def test_unserializable_async_workflow(exporter):
    @atask(name="unserializable_task")
    async def unserializable_task(obj: object):
        return object()

    @aworkflow(name="unserializable_workflow")
    async def unserializable_workflow(obj: object):
        return await unserializable_task(obj)

    await unserializable_task(object())

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["unserializable_task.task"]
