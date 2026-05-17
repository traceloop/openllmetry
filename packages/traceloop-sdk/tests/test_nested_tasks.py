from opentelemetry.semconv_ai import SpanAttributes
from traceloop.sdk.decorators import task, workflow
from pytest import raises


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
    some_workflow_span = spans[3]

    assert (
        inner_inner_task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] ==
        "outer_task.inner_task"
    )
    assert (
        inner_task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == "outer_task"
    )
    with raises(KeyError):
        _ = outer_task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH]
        _ = some_workflow_span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH]
