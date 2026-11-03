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

    # Find the task span (ChatOpenAI span may or may not be present depending on langchain instrumentation)
    task_spans = [span for span in spans if span.name == "answer_question.task"]
    assert len(task_spans) == 1

    task_span = task_spans[0]
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
    dataclass_task(data)

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


def test_json_truncation_with_otel_limit(exporter, monkeypatch):
    """Test that JSON input/output is truncated when OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT is set"""
    # Set environment variable to a small limit for testing
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", "50")

    @task(name="truncation_task")
    def truncation_task(long_input):
        # Return a long output that will also be truncated
        return "This is a very long output string that should definitely exceed the 50 character limit"

    # Call with a long input that will be truncated
    long_input = "This is a very long input string that should definitely exceed the 50 character limit"
    truncation_task(long_input)

    spans = exporter.get_finished_spans()
    task_span = spans[0]

    # Check that input was truncated
    input_json = task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    assert len(input_json) == 50
    assert input_json.startswith('{"args": ["This is a very long input string that s')

    # Check that output was truncated
    output_json = task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    assert len(output_json) == 50
    assert output_json.startswith('"This is a very long output string that should def')


def test_json_no_truncation_without_otel_limit(exporter, monkeypatch):
    """Test that JSON input/output is not truncated when OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT is not set"""
    # Ensure environment variable is not set
    monkeypatch.delenv("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", raising=False)

    @task(name="no_truncation_task")
    def no_truncation_task(long_input):
        return "This is a very long output string that would be truncated if limits were set but should remain intact"

    long_input = "This is a very long input string that would be truncated if limits were set but should remain intact"
    result = no_truncation_task(long_input)

    spans = exporter.get_finished_spans()
    task_span = spans[0]

    # Check that input was not truncated
    input_data = json.loads(task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    assert input_data["args"][0] == long_input

    # Check that output was not truncated
    output_data = json.loads(task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert output_data == result


def test_json_truncation_with_invalid_otel_limit(exporter, monkeypatch):
    """Test that invalid OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT values are ignored"""
    # Set environment variable to invalid value
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", "not_a_number")

    @task(name="invalid_limit_task")
    def invalid_limit_task(test_input):
        return "This output should not be truncated because the limit is invalid"

    test_input = "This input should not be truncated because the limit is invalid"
    result = invalid_limit_task(test_input)

    spans = exporter.get_finished_spans()
    task_span = spans[0]

    # Check that input was not truncated (since invalid limit should be ignored)
    input_data = json.loads(task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    assert input_data["args"][0] == test_input

    # Check that output was not truncated
    output_data = json.loads(task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert output_data == result


@pytest.mark.asyncio
async def test_async_json_truncation_with_otel_limit(exporter, monkeypatch):
    """Test that JSON truncation works with async tasks"""
    # Set environment variable to a small limit for testing
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", "40")

    @task(name="async_truncation_task")
    async def async_truncation_task(long_input):
        await asyncio.sleep(0.1)  # Simulate async work
        return "This is a long async output that should be truncated"

    long_input = "This is a long async input that should be truncated"
    await async_truncation_task(long_input)

    spans = exporter.get_finished_spans()
    task_span = spans[0]

    # Check that input was truncated
    input_json = task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    assert len(input_json) == 40

    # Check that output was truncated
    output_json = task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    assert len(output_json) == 40


def test_json_truncation_preserves_short_content(exporter, monkeypatch):
    """Test that short content is not affected by truncation limits"""
    # Set environment variable to a limit larger than our content
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", "1000")

    @task(name="short_content_task")
    def short_content_task(short_input):
        return "short output"

    short_input = "short input"
    result = short_content_task(short_input)

    spans = exporter.get_finished_spans()
    task_span = spans[0]

    # Check that short input was preserved completely
    input_data = json.loads(task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    assert input_data["args"][0] == short_input

    # Check that short output was preserved completely
    output_data = json.loads(task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert output_data == result


class _UnserializableOnJson:
    """Has a ``.json()`` that raises ``ValueError`` (like a non-JSON ``Response``)."""

    def json(self):
        raise json.JSONDecodeError("Expecting value", "<html>", 0)


class _UnserializableOnToJson:
    def to_json(self):
        raise RuntimeError("to_json failed")


def _circular():
    items = []
    items.append(items)
    return items


UNSERIALIZABLE_VALUES = [
    pytest.param(_UnserializableOnJson, id="json-method-raises-valueerror"),
    pytest.param(_UnserializableOnToJson, id="to_json-raises-runtimeerror"),
    pytest.param(_circular, id="circular-reference"),
]


@pytest.mark.parametrize("make_value", UNSERIALIZABLE_VALUES)
def test_task_input_serialization_failure_does_not_break_call(exporter, make_value):
    @task(name="io_task")
    def io_task(value):
        return "done"

    assert io_task(make_value()) == "done"

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["io_task.task"]
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in spans[0].attributes
    assert spans[0].attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == '"done"'


@pytest.mark.parametrize("make_value", UNSERIALIZABLE_VALUES)
def test_task_output_serialization_failure_does_not_break_call(exporter, make_value):
    value = make_value()

    @task(name="io_task")
    def io_task():
        return value

    assert io_task() is value

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["io_task.task"]
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in spans[0].attributes


@pytest.mark.asyncio
@pytest.mark.parametrize("make_value", UNSERIALIZABLE_VALUES)
async def test_async_task_serialization_failure_does_not_break_call(exporter, make_value):
    value = make_value()

    @task(name="io_task")
    async def io_task(arg):
        return arg

    assert await io_task(value) is value
    assert [span.name for span in exporter.get_finished_spans()] == ["io_task.task"]


@pytest.mark.parametrize("make_value", UNSERIALIZABLE_VALUES)
def test_generator_task_input_serialization_failure_does_not_break_call(exporter, make_value):
    @task(name="io_task")
    def io_task(arg):
        yield 1
        yield 2

    assert list(io_task(make_value())) == [1, 2]
    assert [span.name for span in exporter.get_finished_spans()] == ["io_task.task"]


@pytest.mark.asyncio
@pytest.mark.parametrize("make_value", UNSERIALIZABLE_VALUES)
async def test_async_generator_task_input_serialization_failure_does_not_break_call(
    exporter, make_value
):
    @task(name="io_task")
    async def io_task(arg):
        yield 1
        yield 2

    assert [item async for item in io_task(make_value())] == [1, 2]
    assert [span.name for span in exporter.get_finished_spans()] == ["io_task.task"]


def test_workflow_name_does_not_leak_after_serialization_failure(exporter):
    from traceloop.sdk.decorators import workflow

    @workflow(name="leaky_workflow")
    def leaky_workflow(arg):
        return 1

    @task(name="sibling")
    def sibling():
        return 2

    assert leaky_workflow(_UnserializableOnJson()) == 1
    sibling()

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert set(spans) == {"leaky_workflow.workflow", "sibling.task"}
    assert SpanAttributes.TRACELOOP_WORKFLOW_NAME not in spans["sibling.task"].attributes
