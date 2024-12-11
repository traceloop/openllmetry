import json

from opentelemetry.semconv_ai import SpanAttributes
from traceloop.sdk.decorators import task


def test_instance_method_task(exporter):
    class TestService:
        @task(name="instance_method_task")
        def test_method(self, data: str):
            return f"Processed: {data}"

    service = TestService()
    result = service.test_method("test data")

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["instance_method_task.task"]

    task_span = spans[0]
    assert json.loads(task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "args": ["TestService", "test data"],
        "kwargs": {},
    }
    assert (
        json.loads(task_span.attributes.get(SpanAttributes.TRACELOOP_ENTITY_OUTPUT))
        == result
    )
    assert (
        task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME]
        == "instance_method_task"
    )


def test_class_decorator_task(exporter):
    @task(name="class_decorator_task", method_name="test_method")
    class TestService:
        def test_method(self, data: str):
            return f"Processed: {data}"

    service = TestService()
    result = service.test_method("test data")

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["class_decorator_task.task"]

    task_span = spans[0]
    assert json.loads(task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "args": ["TestService", "test data"],
        "kwargs": {},
    }
    assert (
        json.loads(task_span.attributes.get(SpanAttributes.TRACELOOP_ENTITY_OUTPUT))
        == result
    )
    assert (
        task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME]
        == "class_decorator_task"
    )
