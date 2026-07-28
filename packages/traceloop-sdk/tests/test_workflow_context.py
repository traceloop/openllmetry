"""Regression tests for workflow context scoping."""

import pytest
from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes

from traceloop.sdk.decorators import task, workflow


def _finished_spans_by_name(exporter):
    return {span.name: span for span in exporter.get_finished_spans()}


def _assert_without_workflow(span):
    assert SpanAttributes.TRACELOOP_WORKFLOW_NAME not in span.attributes
    assert span.parent is None


def test_completed_workflow_does_not_tag_following_task(exporter):
    @task(name="inside")
    def inside_task():
        pass

    @workflow(name="first")
    def first_workflow():
        inside_task()

    @task(name="outside")
    def outside_task():
        pass

    first_workflow()
    outside_task()

    spans = _finished_spans_by_name(exporter)
    assert spans["inside.task"].attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "first"
    _assert_without_workflow(spans["outside.task"])


def test_nested_workflow_restores_enclosing_workflow(exporter):
    @task(name="inner_task")
    def inner_task():
        pass

    @workflow(name="inner")
    def inner_workflow():
        inner_task()

    @task(name="after_inner")
    def task_after_inner():
        pass

    @workflow(name="outer")
    def outer_workflow():
        inner_workflow()
        task_after_inner()

    outer_workflow()

    spans = _finished_spans_by_name(exporter)
    assert spans["inner_task.task"].attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "inner"
    assert spans["after_inner.task"].attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "outer"


def test_failed_workflow_does_not_tag_following_task(exporter):
    @workflow(name="failing")
    def failing_workflow():
        raise RuntimeError("expected")

    @task(name="after_failure")
    def task_after_failure():
        pass

    with pytest.raises(RuntimeError, match="expected"):
        failing_workflow()
    task_after_failure()

    spans = _finished_spans_by_name(exporter)
    _assert_without_workflow(spans["after_failure.task"])


@pytest.mark.asyncio
async def test_completed_async_workflow_does_not_tag_following_task(exporter):
    @workflow(name="async_workflow")
    async def async_workflow():
        pass

    @task(name="after_async")
    async def task_after_async():
        pass

    await async_workflow()
    await task_after_async()

    spans = _finished_spans_by_name(exporter)
    _assert_without_workflow(spans["after_async.task"])


def test_generator_workflow_scopes_context_to_iteration(exporter):
    @task(name="inside_stream")
    def task_inside_stream():
        pass

    @task(name="closing_stream")
    def task_while_closing_stream():
        pass

    @workflow(name="stream")
    def stream_workflow():
        try:
            task_inside_stream()
            yield 1
        finally:
            task_while_closing_stream()

    @task(name="before_stream")
    def task_before_stream():
        pass

    @task(name="after_stream")
    def task_after_stream():
        pass

    @task(name="between_stream_items")
    def task_between_stream_items():
        pass

    stream = stream_workflow()
    task_before_stream()
    assert "stream.workflow" not in _finished_spans_by_name(exporter)
    assert next(stream) == 1
    task_between_stream_items()
    stream.close()
    task_after_stream()

    spans = _finished_spans_by_name(exporter)
    _assert_without_workflow(spans["before_stream.task"])
    assert spans["inside_stream.task"].attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "stream"
    assert spans["closing_stream.task"].attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "stream"
    assert spans["closing_stream.task"].parent.span_id == spans["stream.workflow"].context.span_id
    _assert_without_workflow(spans["between_stream_items.task"])
    _assert_without_workflow(spans["after_stream.task"])


@pytest.mark.asyncio
async def test_async_generator_workflow_restores_context(exporter):
    @task(name="inside_async_stream")
    async def task_inside_async_stream():
        pass

    @task(name="closing_async_stream")
    async def task_while_closing_async_stream():
        pass

    @workflow(name="async_stream")
    async def async_stream_workflow():
        try:
            await task_inside_async_stream()
            yield 1
        finally:
            await task_while_closing_async_stream()

    @task(name="after_async_stream")
    async def task_after_async_stream():
        pass

    @task(name="between_async_stream_items")
    async def task_between_async_stream_items():
        pass

    stream = async_stream_workflow()
    result = await anext(stream)
    await task_between_async_stream_items()
    await stream.aclose()
    await task_after_async_stream()

    spans = _finished_spans_by_name(exporter)
    assert result == 1
    assert spans["inside_async_stream.task"].attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "async_stream"
    assert spans["closing_async_stream.task"].attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "async_stream"
    assert spans["closing_async_stream.task"].parent.span_id == spans["async_stream.workflow"].context.span_id
    _assert_without_workflow(spans["between_async_stream_items.task"])
    _assert_without_workflow(spans["after_async_stream.task"])


def test_generator_preserves_inner_span_across_yield(exporter):
    tracer = trace.get_tracer(__name__)

    @workflow(name="stream")
    def stream_workflow():
        with tracer.start_as_current_span("long_child"):
            yield 1
            with tracer.start_as_current_span("after_yield"):
                pass

    stream = stream_workflow()
    assert next(stream) == 1
    with pytest.raises(StopIteration):
        next(stream)

    spans = _finished_spans_by_name(exporter)
    assert spans["after_yield"].parent.span_id == spans["long_child"].context.span_id


@pytest.mark.asyncio
async def test_async_generator_preserves_inner_span_across_yield(exporter):
    tracer = trace.get_tracer(__name__)

    @workflow(name="async_stream")
    async def async_stream_workflow():
        with tracer.start_as_current_span("async_long_child"):
            yield 1
            with tracer.start_as_current_span("async_after_yield"):
                pass

    stream = async_stream_workflow()
    assert await anext(stream) == 1
    with pytest.raises(StopAsyncIteration):
        await anext(stream)

    spans = _finished_spans_by_name(exporter)
    assert spans["async_after_yield"].parent.span_id == spans["async_long_child"].context.span_id


def test_generator_task_propagates_entity_path(exporter):
    tracer = trace.get_tracer(__name__)

    @task(name="stream_task")
    def stream_task():
        with tracer.start_as_current_span("stream_child"):
            pass
        yield 1

    assert list(stream_task()) == [1]

    spans = _finished_spans_by_name(exporter)
    assert spans["stream_child"].attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == "stream_task"
