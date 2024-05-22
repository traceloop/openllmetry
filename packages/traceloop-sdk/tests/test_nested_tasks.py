import pytest
from opentelemetry.semconv.ai import SpanAttributes
from traceloop.sdk.decorators import task, workflow


@pytest.mark.vcr
def test_nested_tasks(exporter):
    @workflow(name="some_workflow")
    def some_workflow():
        return outer_task()

    @task(name="outer_task")
    def outer_task():
        return inner_task()

    @task(name="inner_task")
    def inner_task():
        return inner_inner_task()

    @task(name="inner_inner_task")
    def inner_inner_task():
        return

    some_workflow()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "inner_inner_task.task",
        "inner_task.task",
        "outer_task.task",
        "some_workflow.workflow",
    ]

    inner_inner_task_span = spans[0]
    inner_task_span = spans[1]
    outer_task_span = spans[2]
    assert inner_inner_task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "outer_task.inner_task.inner_inner_task"
    assert inner_task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "outer_task.inner_task"
    assert outer_task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "outer_task"
