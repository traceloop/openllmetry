import pytest
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_CONVERSATION_ID,
)
from opentelemetry.semconv_ai import SpanAttributes
from traceloop.sdk.tracing import set_conversation_id
from traceloop.sdk.decorators import conversation, task, workflow


def test_set_conversation_id_api(exporter):
    """Test set_conversation_id() API"""

    @workflow(name="chat_workflow")
    def chat_workflow(conv_id: str):
        set_conversation_id(conv_id)
        return test_task()

    @task(name="test_task")
    def test_task():
        return "response"

    chat_workflow("conv-abc")

    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    # The workflow span is created before set_conversation_id is called,
    # so it won't have conversation_id. But the task span should have it.
    task_span = next(span for span in spans if "task" in span.name)
    assert task_span.attributes[GEN_AI_CONVERSATION_ID] == "conv-abc"


def test_conversation_decorator_sync(exporter):
    """Test conversation decorator with sync function"""

    @conversation(conversation_id="conv-123")
    @workflow(name="test_conversation")
    def chat_session(message: str):
        return inner_task(message)

    @task(name="inner_task")
    def inner_task(message: str):
        return f"Response to: {message}"

    result = chat_session("Hello")

    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    # All spans should have conversation_id
    for span in spans:
        assert span.attributes[GEN_AI_CONVERSATION_ID] == "conv-123"

    # Workflow span should have correct name
    workflow_span = next(span for span in spans if "workflow" in span.name)
    assert workflow_span.attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "test_conversation"


@pytest.mark.asyncio
async def test_conversation_decorator_async(exporter):
    """Test conversation decorator with async function"""

    @conversation(conversation_id="conv-456")
    @workflow(name="async_conversation")
    async def async_chat_session(message: str):
        return await async_inner_task(message)

    @task(name="async_inner_task")
    async def async_inner_task(message: str):
        return f"Response to: {message}"

    result = await async_chat_session("Hello")

    spans = exporter.get_finished_spans()

    # All spans should have conversation_id
    for span in spans:
        assert span.attributes[GEN_AI_CONVERSATION_ID] == "conv-456"


def test_conversation_id_in_nested_tasks(exporter):
    """Test that conversation_id propagates through nested tasks"""

    @workflow(name="chat_workflow")
    def chat_workflow():
        set_conversation_id("conv-nested-789")
        return level1_task()

    @task(name="level1_task")
    def level1_task():
        return level2_task()

    @task(name="level2_task")
    def level2_task():
        return "deep response"

    chat_workflow()

    spans = exporter.get_finished_spans()
    assert len(spans) == 3

    # Task spans should have conversation_id (workflow span won't because it's created before set_conversation_id)
    task_spans = [span for span in spans if "task" in span.name]
    assert len(task_spans) == 2
    for span in task_spans:
        assert span.attributes[GEN_AI_CONVERSATION_ID] == "conv-nested-789"


def test_conversation_id_with_multiple_calls(exporter):
    """Test that conversation_id works correctly with multiple separate calls"""

    @workflow(name="chat_workflow")
    def chat_workflow(conv_id: str):
        set_conversation_id(conv_id)
        return test_task()

    @task(name="test_task")
    def test_task():
        return "response"

    # First conversation
    chat_workflow("conv-first")
    spans_first = exporter.get_finished_spans()
    exporter.clear()

    # Second conversation
    chat_workflow("conv-second")
    spans_second = exporter.get_finished_spans()

    # Verify first conversation task spans have correct conversation_id
    task_span_first = next(span for span in spans_first if "task" in span.name)
    assert task_span_first.attributes[GEN_AI_CONVERSATION_ID] == "conv-first"

    # Verify second conversation task spans have correct conversation_id
    task_span_second = next(span for span in spans_second if "task" in span.name)
    assert task_span_second.attributes[GEN_AI_CONVERSATION_ID] == "conv-second"


def test_conversation_decorator_standalone(exporter):
    """Test conversation decorator without workflow"""

    @conversation(conversation_id="conv-standalone")
    @task(name="chat_task")
    def my_chat_function():
        return "response"

    my_chat_function()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1

    # Should have conversation_id
    assert spans[0].attributes[GEN_AI_CONVERSATION_ID] == "conv-standalone"
