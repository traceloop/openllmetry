import pytest
from traceloop.sdk import Traceloop, AssociationProperty
from traceloop.sdk.decorators import task, workflow
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes


@pytest.fixture
def client_with_exporter():
    """
    Fixture that initializes Traceloop with API key.
    Client is only created when NO custom exporter/processor is provided.
    """
    # Initialize with API key and Traceloop endpoint - this creates a client
    client = Traceloop.init(
        app_name="test_associations",
        api_key="test-api-key",
        api_endpoint="https://api.traceloop.com",
        disable_batch=True,
        # NO exporter or processor - so client gets created
    )

    # Get spans from the tracer provider for assertions
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    tracer_provider = trace.get_tracer_provider()
    if isinstance(tracer_provider, TracerProvider):
        # Get the span processor's exporter
        span_processors = tracer_provider._active_span_processor._span_processors
        # Find the exporter from the processors
        for processor in span_processors:
            if hasattr(processor, 'span_exporter'):
                exporter = processor.span_exporter
                break
        else:
            # Fallback: create a mock exporter
            exporter = InMemorySpanExporter()
    else:
        exporter = InMemorySpanExporter()

    yield client, exporter

    # Cleanup
    if hasattr(exporter, 'clear'):
        exporter.clear()


def test_associations_create_single(client_with_exporter):
    """Test creating a single association."""
    client, exporter = client_with_exporter

    @workflow(name="test_single_association")
    def test_workflow():
        return test_task()

    @task(name="test_single_task")
    def test_task():
        return

    client.associations.set([(AssociationProperty.SESSION_ID, "conv-123")])
    test_workflow()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "test_single_task.task",
        "test_single_association.workflow",
    ]

    task_span = spans[0]
    workflow_span = spans[1]

    assert workflow_span.attributes[
        f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.session_id"
    ] == "conv-123"
    assert task_span.attributes[
        f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.session_id"
    ] == "conv-123"


def test_associations_create_multiple(client_with_exporter):
    """Test creating multiple associations at once."""
    client, exporter = client_with_exporter

    @workflow(name="test_multiple_associations")
    def test_workflow():
        return test_task()

    @task(name="test_multiple_task")
    def test_task():
        return

    client.associations.set([
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
    assert workflow_span.attributes[
        f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"
    ] == "user-456"
    assert workflow_span.attributes[
        f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.session_id"
    ] == "session-789"
    assert workflow_span.attributes[
        f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.customer_id"
    ] == "customer-999"

    assert task_span.attributes[
        f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"
    ] == "user-456"
    assert task_span.attributes[
        f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.session_id"
    ] == "session-789"
    assert task_span.attributes[
        f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.customer_id"
    ] == "customer-999"


def test_associations_within_workflow(client_with_exporter):
    """Test creating associations within a workflow."""
    client, exporter = client_with_exporter

    @workflow(name="test_associations_within")
    def test_workflow():
        client.associations.set([
            (AssociationProperty.SESSION_ID, "conv-abc"),
            (AssociationProperty.USER_ID, "user-xyz"),
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
    assert workflow_span.attributes[
        f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.session_id"
    ] == "conv-abc"
    assert workflow_span.attributes[
        f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"
    ] == "user-xyz"

    assert task_span.attributes[
        f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.session_id"
    ] == "conv-abc"
    assert task_span.attributes[
        f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"
    ] == "user-xyz"


def test_all_association_properties(client_with_exporter):
    """Test that all AssociationProperty enum values work correctly."""
    client, exporter = client_with_exporter

    @workflow(name="test_all_properties")
    def test_workflow():
        return

    client.associations.set([
        (AssociationProperty.SESSION_ID, "conv-1"),
        (AssociationProperty.CUSTOMER_ID, "customer-2"),
        (AssociationProperty.USER_ID, "user-3"),
        (AssociationProperty.SESSION_ID, "session-4"),
    ])
    test_workflow()

    spans = exporter.get_finished_spans()
    workflow_span = spans[0]

    assert workflow_span.attributes[
        f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.customer_id"
    ] == "customer-2"
    assert workflow_span.attributes[
        f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"
    ] == "user-3"
    assert workflow_span.attributes[
        f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.session_id"
    ] == "session-4"
