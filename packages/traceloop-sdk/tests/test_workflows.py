import asyncio
import json
import dataclasses

import pytest
from openai import OpenAI, AsyncOpenAI
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace.status import StatusCode
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task


@pytest.fixture
def openai_client():
    return OpenAI()


@pytest.fixture
def async_openai_client():
    return AsyncOpenAI()


@pytest.mark.vcr
def test_simple_workflow(exporter, openai_client):
    @task(name="something_creator", version=2)
    def create_something(what: str, subject: str):
        Traceloop.set_prompt("Tell me a {what} about {subject}", {"what": what, "subject": subject}, 5)
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
    open_ai_span = next(span for span in spans if span.name == "openai.chat")
    assert open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"] == "Tell me a joke about OpenTelemetry"
    assert open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    assert open_ai_span.attributes.get("traceloop.prompt.template") == "Tell me a {what} about {subject}"
    assert open_ai_span.attributes.get("traceloop.prompt.template_variables.what") == "joke"
    assert open_ai_span.attributes.get("traceloop.prompt.template_variables.subject") == "OpenTelemetry"
    assert open_ai_span.attributes.get("traceloop.prompt.version") == "5"

    workflow_span = next(span for span in spans if span.name == "pirate_joke_generator.workflow")
    task_span = next(span for span in spans if span.name == "something_creator.task")
    assert json.loads(task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "args": ["joke"],
        "kwargs": {"subject": "OpenTelemetry"},
    }

    assert json.loads(task_span.attributes.get(SpanAttributes.TRACELOOP_ENTITY_OUTPUT)) == joke
    assert task_span.parent.span_id == workflow_span.context.span_id
    assert workflow_span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "pirate_joke_generator"
    assert workflow_span.attributes[SpanAttributes.TRACELOOP_ENTITY_VERSION] == 1
    assert task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_VERSION] == 2


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_simple_aworkflow(exporter, async_openai_client):
    @task(name="something_creator", version=2)
    async def create_something(what: str, subject: str):
        Traceloop.set_prompt("Tell me a {what} about {subject}", {"what": what, "subject": subject}, 5)
        completion = await async_openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Tell me a {what} about {subject}"}],
        )
        return completion.choices[0].message.content

    @workflow(name="pirate_joke_generator", version=1)
    async def joke_workflow():
        return await create_something("joke", subject="OpenTelemetry")

    joke = await joke_workflow()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
        "something_creator.task",
        "pirate_joke_generator.workflow",
    ]
    open_ai_span = next(span for span in spans if span.name == "openai.chat")
    assert open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"] == "Tell me a joke about OpenTelemetry"
    assert open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    assert open_ai_span.attributes.get("traceloop.prompt.template") == "Tell me a {what} about {subject}"
    assert open_ai_span.attributes.get("traceloop.prompt.template_variables.what") == "joke"
    assert open_ai_span.attributes.get("traceloop.prompt.template_variables.subject") == "OpenTelemetry"
    assert open_ai_span.attributes.get("traceloop.prompt.version") == "5"

    workflow_span = next(span for span in spans if span.name == "pirate_joke_generator.workflow")
    task_span = next(span for span in spans if span.name == "something_creator.task")
    assert json.loads(task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "args": ["joke"],
        "kwargs": {"subject": "OpenTelemetry"},
    }

    assert json.loads(task_span.attributes.get(SpanAttributes.TRACELOOP_ENTITY_OUTPUT)) == joke
    assert task_span.parent.span_id == workflow_span.context.span_id
    assert workflow_span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "pirate_joke_generator"
    assert workflow_span.attributes[SpanAttributes.TRACELOOP_ENTITY_VERSION] == 1
    assert task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_VERSION] == 2


@pytest.mark.vcr
def test_streaming_workflow(exporter, openai_client):
    @task(name="pirate_joke_generator")
    def joke_task():
        response_stream = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Tell me a joke about OpenTelemetry"}],
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
    generator_span = next(span for span in spans if span.name == "pirate_joke_generator.task")
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

    assert workflow_1_span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "workflow_1"
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
    @task(name="unserializable_task")
    async def unserializable_task(obj: object):
        return object()

    @workflow(name="unserializable_workflow")
    async def unserializable_workflow(obj: object):
        return await unserializable_task(obj)

    await unserializable_task(object())

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["unserializable_task.task"]


@pytest.mark.asyncio
async def test_async_generator_workflow(exporter):
    @workflow(name="async generator")
    async def stream_numbers():
        for i in range(3):
            yield i
            await asyncio.sleep(0.1)

    results = []

    async for num in stream_numbers():
        results.append(num)

    spans = exporter.get_finished_spans()

    assert results == [0, 1, 2]
    assert [span.name for span in spans] == ["async generator.workflow"]


def test_sync_workflow_error_handling(exporter):
    """Test error handling in synchronous workflow"""
    error_message = "Intentional workflow test error"

    @workflow(name="failing_workflow")
    def failing_workflow():
        raise ValueError(error_message)

    with pytest.raises(ValueError, match=error_message):
        failing_workflow()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1

    workflow_span = spans[0]
    assert workflow_span.name == "failing_workflow.workflow"
    assert workflow_span.status.status_code == StatusCode.ERROR
    assert error_message in workflow_span.status.description


@pytest.mark.asyncio
async def test_async_workflow_error_handling(exporter):
    """Test error handling in asynchronous workflow"""
    error_message = "Intentional async workflow test error"

    @workflow(name="failing_async_workflow")
    async def failing_async_workflow():
        await asyncio.sleep(0.1)
        raise ValueError(error_message)

    with pytest.raises(ValueError, match=error_message):
        await failing_async_workflow()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1

    workflow_span = spans[0]
    assert workflow_span.name == "failing_async_workflow.workflow"
    assert workflow_span.status.status_code == StatusCode.ERROR
    assert error_message in workflow_span.status.description


def test_sync_generator_workflow_error_handling(exporter):
    """Test error handling in synchronous generator workflow"""
    error_message = "Intentional generator workflow test error"

    @workflow(name="failing_generator_workflow")
    def failing_generator_workflow():
        yield 1
        yield 2
        raise ValueError(error_message)

    results = []
    with pytest.raises(ValueError, match=error_message):
        for num in failing_generator_workflow():
            results.append(num)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1

    workflow_span = spans[0]
    assert workflow_span.name == "failing_generator_workflow.workflow"
    assert workflow_span.status.status_code == StatusCode.ERROR
    assert error_message in workflow_span.status.description
    assert results == [1, 2]


@pytest.mark.asyncio
async def test_async_generator_workflow_error_handling(exporter):
    """Test error handling in asynchronous generator workflow"""
    error_message = "Intentional async generator workflow test error"

    @workflow(name="failing_async_generator_workflow")
    async def failing_async_generator_workflow():
        yield 1
        await asyncio.sleep(0.1)
        yield 2
        await asyncio.sleep(0.1)
        raise ValueError(error_message)

    results = []
    with pytest.raises(ValueError, match=error_message):
        async for num in failing_async_generator_workflow():
            results.append(num)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1

    workflow_span = spans[0]
    assert workflow_span.name == "failing_async_generator_workflow.workflow"
    assert workflow_span.status.status_code == StatusCode.ERROR
    assert error_message in workflow_span.status.description
    assert results == [1, 2]


def test_nested_error_handling(exporter):
    """Test error handling in a workflow that calls a task that raises an error"""
    error_message = "Intentional nested error"

    @task(name="failing_nested_task")
    def failing_nested_task():
        raise ValueError(error_message)

    @workflow(name="parent_workflow")
    def parent_workflow():
        return failing_nested_task()

    with pytest.raises(ValueError, match=error_message):
        parent_workflow()

    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    # Check task span
    task_span = next(span for span in spans if span.name == "failing_nested_task.task")
    assert task_span.status.status_code == StatusCode.ERROR
    assert error_message in task_span.status.description

    # Check workflow span
    workflow_span = next(span for span in spans if span.name == "parent_workflow.workflow")
    assert workflow_span.status.status_code == StatusCode.ERROR
    assert error_message in workflow_span.status.description


@pytest.mark.asyncio
async def test_nested_async_error_handling(exporter):
    """Test error handling in an async workflow that calls an async task that raises an error"""
    error_message = "Intentional nested async error"

    @task(name="failing_nested_async_task")
    async def failing_nested_async_task():
        await asyncio.sleep(0.1)
        raise ValueError(error_message)

    @workflow(name="parent_async_workflow")
    async def parent_async_workflow():
        return await failing_nested_async_task()

    with pytest.raises(ValueError, match=error_message):
        await parent_async_workflow()

    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    # Check task span
    task_span = next(span for span in spans if span.name == "failing_nested_async_task.task")
    assert task_span.status.status_code == StatusCode.ERROR
    assert error_message in task_span.status.description

    # Check workflow span
    workflow_span = next(span for span in spans if span.name == "parent_async_workflow.workflow")
    assert workflow_span.status.status_code == StatusCode.ERROR
    assert error_message in workflow_span.status.description


def test_nested_generator_error_handling(exporter):
    """Test error handling in a workflow that calls a generator task that raises an error"""
    error_message = "Intentional nested generator error"

    @task(name="failing_generator_task")
    def failing_generator_task():
        yield 1
        yield 2
        raise ValueError(error_message)

    @workflow(name="generator_parent_workflow")
    def generator_parent_workflow():
        results = []
        for num in failing_generator_task():
            results.append(num)
        return results

    with pytest.raises(ValueError, match=error_message):
        generator_parent_workflow()

    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    # Check task span
    task_span = next(span for span in spans if span.name == "failing_generator_task.task")
    assert task_span.status.status_code == StatusCode.ERROR
    assert error_message in task_span.status.description

    # Check workflow span
    workflow_span = next(span for span in spans if span.name == "generator_parent_workflow.workflow")
    assert workflow_span.status.status_code == StatusCode.ERROR
    assert error_message in workflow_span.status.description


@pytest.mark.asyncio
async def test_nested_async_generator_error_handling(exporter):
    """Test error handling in an async workflow that calls an async generator task that raises an error"""
    error_message = "Intentional nested async generator error"

    @task(name="failing_async_generator_task")
    async def failing_async_generator_task():
        yield 1
        await asyncio.sleep(0.1)
        yield 2
        await asyncio.sleep(0.1)
        raise ValueError(error_message)

    @workflow(name="async_generator_parent_workflow")
    async def async_generator_parent_workflow():
        results = []
        async for num in failing_async_generator_task():
            results.append(num)
        return results

    with pytest.raises(ValueError, match=error_message):
        await async_generator_parent_workflow()

    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    # Check task span
    task_span = next(span for span in spans if span.name == "failing_async_generator_task.task")
    assert task_span.status.status_code == StatusCode.ERROR
    assert error_message in task_span.status.description

    # Check workflow span
    workflow_span = next(span for span in spans if span.name == "async_generator_parent_workflow.workflow")
    assert workflow_span.status.status_code == StatusCode.ERROR
    assert error_message in workflow_span.status.description


@dataclasses.dataclass
class TestDataClass:
    field1: str
    field2: int


def test_dataclass_serialization_workflow(exporter):
    @task(name="dataclass_task")
    def dataclass_task(data: TestDataClass):
        return data

    @workflow(name="dataclass_workflow")
    def dataclass_workflow(data: TestDataClass):
        return dataclass_task(data)

    data = TestDataClass(field1="value1", field2=123)
    dataclass_workflow(data)

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["dataclass_task.task", "dataclass_workflow.workflow"]

    task_span = spans[0]
    workflow_span = spans[1]

    assert json.loads(task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "args": [{"field1": "value1", "field2": 123}],
        "kwargs": {},
    }
    assert json.loads(task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "field1": "value1",
        "field2": 123,
    }
    assert task_span.parent.span_id == workflow_span.context.span_id
    assert workflow_span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "dataclass_workflow"
    assert task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "dataclass_task"
