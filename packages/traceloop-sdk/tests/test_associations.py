from traceloop.sdk import Traceloop
from traceloop.sdk.tracing.associations import AssociationProperty
from traceloop.sdk.decorators import task, workflow


def test_associations_create_single(exporter):
    """Test creating a single association."""
    @workflow(name="test_single_association")
    def test_workflow():
        return test_task()

    @task(name="test_single_task")
    def test_task():
        return

    Traceloop.associations.set([(AssociationProperty.CONVERSATION_ID, "conv-123")])
    test_workflow()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "test_single_task.task",
        "test_single_association.workflow",
    ]

    task_span = spans[0]
    workflow_span = spans[1]

    assert workflow_span.attributes["conversation_id"] == "conv-123"
    assert task_span.attributes["conversation_id"] == "conv-123"


def test_associations_create_multiple(exporter):
    """Test creating multiple associations at once."""
    @workflow(name="test_multiple_associations")
    def test_workflow():
        return test_task()

    @task(name="test_multiple_task")
    def test_task():
        return

    Traceloop.associations.set([
        (AssociationProperty.USER_ID, "user-456"),
        (AssociationProperty.SESSION_ID, "session-789"),
        (AssociationProperty.CUSTOMER_ID, "customer-999"),
    ])
    test_workflow()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "test_multiple_task.task",
        "test_multiple_associations.workflow",
    ]

    task_span = spans[0]
    workflow_span = spans[1]

    # Check all associations are present
    assert workflow_span.attributes["user_id"] == "user-456"
    assert workflow_span.attributes["session_id"] == "session-789"
    assert workflow_span.attributes["customer_id"] == "customer-999"

    assert task_span.attributes["user_id"] == "user-456"
    assert task_span.attributes["session_id"] == "session-789"
    assert task_span.attributes["customer_id"] == "customer-999"


def test_associations_within_workflow(exporter):
    """Test creating associations within a workflow."""
    @workflow(name="test_associations_within")
    def test_workflow():
        Traceloop.associations.set([
            (AssociationProperty.CONVERSATION_ID, "conv-abc"),
            (AssociationProperty.THREAD_ID, "thread-xyz"),
        ])
        return test_task()

    @task(name="test_within_task")
    def test_task():
        return

    test_workflow()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "test_within_task.task",
        "test_associations_within.workflow",
    ]

    task_span = spans[0]
    workflow_span = spans[1]

    # Both spans should have all associations
    assert workflow_span.attributes["conversation_id"] == "conv-abc"
    assert workflow_span.attributes["thread_id"] == "thread-xyz"

    assert task_span.attributes["conversation_id"] == "conv-abc"
    assert task_span.attributes["thread_id"] == "thread-xyz"


def test_all_association_properties(exporter):
    """Test that all AssociationProperty enum values work correctly."""
    @workflow(name="test_all_properties")
    def test_workflow():
        return

    Traceloop.associations.set([
        (AssociationProperty.CONVERSATION_ID, "conv-1"),
        (AssociationProperty.CUSTOMER_ID, "customer-2"),
        (AssociationProperty.USER_ID, "user-3"),
        (AssociationProperty.SESSION_ID, "session-4"),
        (AssociationProperty.THREAD_ID, "thread-5"),
    ])
    test_workflow()

    spans = exporter.get_finished_spans()
    workflow_span = spans[0]

    assert workflow_span.attributes["conversation_id"] == "conv-1"
    assert workflow_span.attributes["customer_id"] == "customer-2"
    assert workflow_span.attributes["user_id"] == "user-3"
    assert workflow_span.attributes["session_id"] == "session-4"
    assert workflow_span.attributes["thread_id"] == "thread-5"
