import json
import pytest
import asyncio
import dataclasses
from typing import Generator, AsyncGenerator

from langchain_openai import ChatOpenAI
from traceloop.sdk.decorators import task
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace.status import StatusCode


@pytest.mark.vcr
def test_task_io_serialization_with_langchain(exporter):
    @task(name="answer_question")
    def answer_question():
        chat = ChatOpenAI(temperature=0)

        return chat.invoke("Is Berlin the capital of Germany? Answer with yes or no")

    answer_question()

    spans = exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "ChatOpenAI.chat",
        "answer_question.task",
    ]

    task_span = next(span for span in spans if span.name == "answer_question.task")
    assert json.loads(task_span.attributes.get(SpanAttributes.TRACELOOP_ENTITY_OUTPUT))["kwargs"]["content"] == "Yes"


def test_sync_task_error_handling(exporter):
    """Test error handling in synchronous task"""
    error_message = "Intentional test error"

    @task(name="failing_task")
    def failing_task():
        raise ValueError(error_message)

    with pytest.raises(ValueError, match=error_message):
        failing_task()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1

    task_span = spans[0]
    assert task_span.name == "failing_task.task"
    assert task_span.status.status_code == StatusCode.ERROR
    assert error_message in task_span.status.description


@pytest.mark.asyncio
async def test_async_task_error_handling(exporter):
    """Test error handling in asynchronous task"""
    error_message = "Intentional async test error"

    @task(name="failing_async_task")
    async def failing_async_task() -> None:
        await asyncio.sleep(0.1)
        raise ValueError(error_message)

    with pytest.raises(ValueError, match=error_message):
        await failing_async_task()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1

    task_span = spans[0]
    assert task_span.name == "failing_async_task.task"
    assert task_span.status.status_code == StatusCode.ERROR
    assert error_message in task_span.status.description


def test_sync_generator_task_error_handling(exporter):
    """Test error handling in synchronous generator task"""
    error_message = "Intentional generator test error"

    @task(name="failing_generator_task")
    def failing_generator_task() -> Generator[int, None, None]:
        yield 1
        yield 2
        raise ValueError(error_message)

    results = []
    with pytest.raises(ValueError, match=error_message):
        for num in failing_generator_task():
            results.append(num)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1

    task_span = spans[0]
    assert task_span.name == "failing_generator_task.task"
    assert task_span.status.status_code == StatusCode.ERROR
    assert error_message in task_span.status.description
    assert results == [1, 2]


@pytest.mark.asyncio
async def test_async_generator_task_error_handling(exporter):
    """Test error handling in asynchronous generator task"""
    error_message = "Intentional async generator test error"

    @task(name="failing_async_generator_task")
    async def failing_async_generator_task() -> AsyncGenerator[int, None]:
        yield 1
        await asyncio.sleep(0.1)
        raise ValueError(error_message)
        yield 2
        await asyncio.sleep(0.1)

    results = []
    with pytest.raises(ValueError, match=error_message):
        async for num in failing_async_generator_task():
            results.append(num)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1

    task_span = spans[0]
    assert task_span.name == "failing_async_generator_task.task"
    assert task_span.status.status_code == StatusCode.ERROR
    assert error_message in task_span.status.description
    assert results == [1]


@dataclasses.dataclass
class TestDataClass:
    field1: str
    field2: int


def test_dataclass_serialization_task(exporter):
    @task(name="dataclass_task")
    def dataclass_task(data: TestDataClass):
        return data

    data = TestDataClass(field1="value1", field2=123)
    result = dataclass_task(data)

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["dataclass_task.task"]

    task_span = spans[0]

    assert json.loads(task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "args": [{"field1": "value1", "field2": 123}],
        "kwargs": {},
    }
    assert json.loads(task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "field1": "value1",
        "field2": 123,
    }
