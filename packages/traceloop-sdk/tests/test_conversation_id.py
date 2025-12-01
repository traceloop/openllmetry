from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow


def test_set_conversation_id(exporter):
    """Test that conversation_id is set on all spans when called before workflow."""
    @workflow(name="test_conversation_workflow")
    def test_workflow():
        return test_task()

    @task(name="test_conversation_task")
    def test_task():
        return

    Traceloop.set_conversation_id("conv-12345")
    test_workflow()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "test_conversation_task.task",
        "test_conversation_workflow.workflow",
    ]

    task_span = spans[0]
    workflow_span = spans[1]

    # Check that conversation_id is set directly without the prefix
    assert workflow_span.attributes["conversation_id"] == "conv-12345"
    assert task_span.attributes["conversation_id"] == "conv-12345"


def test_set_conversation_id_within_workflow(exporter):
    """Test that conversation_id is set when called within a workflow."""
    @workflow(name="test_conversation_within_workflow")
    def test_workflow():
        Traceloop.set_conversation_id("conv-67890")
        return test_task()

    @task(name="test_conversation_within_task")
    def test_task():
        return

    test_workflow()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "test_conversation_within_task.task",
        "test_conversation_within_workflow.workflow",
    ]

    task_span = spans[0]
    workflow_span = spans[1]

    # Both spans should have conversation_id set
    assert workflow_span.attributes["conversation_id"] == "conv-67890"
    assert task_span.attributes["conversation_id"] == "conv-67890"
