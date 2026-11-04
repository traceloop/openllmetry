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

    chat_session("Hello")

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

    await async_chat_session("Hello")

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


def test_conversation_decorator_sync_generator(exporter):
    """@conversation on a generator must apply the id to spans produced during
    iteration, not detach it before the body runs."""

    @conversation(conversation_id="conv-gen")
    def streaming_chat():
        yield inner_task("a")
        yield inner_task("b")

    @task(name="inner_task")
    def inner_task(msg: str):
        return msg

    # Fully consume the generator so its body (and the task spans) actually run.
    assert list(streaming_chat()) == ["a", "b"]

    task_spans = [s for s in exporter.get_finished_spans() if "task" in s.name]
    assert len(task_spans) == 2
    for span in task_spans:
        assert span.attributes[GEN_AI_CONVERSATION_ID] == "conv-gen"


@pytest.mark.asyncio
async def test_conversation_decorator_async_generator(exporter):
    """Same guarantee for async generators (which are NOT coroutine functions, so
    they must not fall into a branch that detaches before iteration)."""

    @conversation(conversation_id="conv-agen")
    async def streaming_chat():
        yield await inner_task("a")
        yield await inner_task("b")

    @task(name="inner_task")
    async def inner_task(msg: str):
        return msg

    collected = [item async for item in streaming_chat()]
    assert collected == ["a", "b"]

    task_spans = [s for s in exporter.get_finished_spans() if "task" in s.name]
    assert len(task_spans) == 2
    for span in task_spans:
        assert span.attributes[GEN_AI_CONVERSATION_ID] == "conv-agen"


@pytest.mark.asyncio
async def test_conversation_decorator_async_detach_does_not_crash(exporter, monkeypatch):
    """Detach on the @conversation async path must not crash user code when the
    coroutine resumes on a different context (the cross-Context ValueError case)."""
    from opentelemetry import context as context_api

    # Simulate the cross-task/thread resume: detach raises the ValueError that OTel
    # raises for a token created in a different Context.
    def raising_detach(token):
        raise ValueError("was created in a different Context")

    monkeypatch.setattr(context_api, "detach", raising_detach)

    @conversation(conversation_id="conv-crash")
    async def handler():
        return "ok"

    # Must return normally, not surface the detach ValueError.
    assert await handler() == "ok"


def test_conversation_id_does_not_leak_to_later_sibling(exporter):
    """conversation_id is scoped to the decorated function, not the rest of the thread.

    The @conversation decorator attaches conversation_id to the OTel context. If the
    token is never detached, the id sticks after the decorated function returns and
    leaks onto unrelated work that runs later on the same thread.
    """

    @conversation(conversation_id="conv-scoped")
    @workflow(name="chat_session")
    def chat_session():
        return "answer"

    @workflow(name="unrelated_later")
    def unrelated_later():
        return "other"

    chat_session()
    unrelated_later()

    by_name = {span.name: span for span in exporter.get_finished_spans()}

    # The decorated function's own span carries the id (that is the contract).
    assert (
        by_name["chat_session.workflow"].attributes[GEN_AI_CONVERSATION_ID]
        == "conv-scoped"
    )
    # The crux: work that runs AFTER the @conversation function returned must not
    # inherit its conversation_id.
    assert (
        GEN_AI_CONVERSATION_ID not in by_name["unrelated_later.workflow"].attributes
    )


def test_conversation_id_survives_generator_returned_by_inner_decorator(exporter):
    """@conversation above @workflow on a GENERATOR function keeps the id attached.

    inspect.isgeneratorfunction is False for base.sync_wrap -- it is a plain
    function that RETURNS a generator -- so @conversation takes its sync branch.
    Detaching when that call returns would drop conversation_id before a single
    item is produced, leaving every span created during iteration unlabelled.
    The sync wrapper therefore hands the token to the returned iterator.
    """

    @task(name="first_step")
    def first_step():
        return "1"

    @task(name="second_step")
    def second_step():
        return "2"

    @conversation(conversation_id="conv-generator")
    @workflow(name="streaming_chat")
    def streaming_chat():
        first_step()
        yield "a"
        second_step()
        yield "b"

    assert list(streaming_chat()) == ["a", "b"]

    by_name = {span.name: span for span in exporter.get_finished_spans()}
    # Spans created DURING iteration must carry the conversation id.
    assert (
        by_name["first_step.task"].attributes[GEN_AI_CONVERSATION_ID]
        == "conv-generator"
    )
    assert (
        by_name["second_step.task"].attributes[GEN_AI_CONVERSATION_ID]
        == "conv-generator"
    )


def test_conversation_id_released_when_generator_abandoned(exporter):
    """Handing the token to the iterator must not turn into a leak.

    If the consumer stops early, GeneratorExit still runs the iterator's finally,
    so the id is released rather than sticking on the context.
    """
    from opentelemetry import context as context_api

    @conversation(conversation_id="conv-abandoned")
    @workflow(name="abandoned_stream")
    def abandoned_stream():
        yield "a"
        yield "b"

    for _ in abandoned_stream():
        break

    assert context_api.get_value("conversation_id") is None
