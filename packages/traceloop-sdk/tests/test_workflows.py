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

    @workflow(name="pirate_joke_generator", version=1)
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

    @task(name="pirate_joke_generator")
    def joke_task():
        response_stream = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Tell me a joke about OpenTelemetry"}
            ],
            stream=True,
        )
        for chunk in response_stream:
            yield chunk

    @task(name="joke_runner")
    def joke_runner():
        res = joke_task()
        return res

    @workflow(name="joke_manager")
    def joke_workflow():
        res = joke_runner()
        for chunk in res:
            pass

    joke_workflow()

    spans = exporter.get_finished_spans()
    assert set([span.name for span in spans]) == set(
        [
            "openai.chat",
            "pirate_joke_generator.task",
            "joke_runner.task",
            "joke_manager.workflow",
        ]
    )
    generator_span = next(
        span for span in spans if span.name == "pirate_joke_generator.task"
    )
    runner_span = next(span for span in spans if span.name == "joke_runner.task")
    manager_span = next(span for span in spans if span.name == "joke_manager.workflow")
    openai_span = next(span for span in spans if span.name == "openai.chat")

    assert openai_span.parent.span_id == generator_span.context.span_id
    assert generator_span.parent.span_id == runner_span.context.span_id
    assert runner_span.parent.span_id == manager_span.context.span_id
    assert openai_span.end_time <= manager_span.end_time


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
