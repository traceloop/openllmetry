import asyncio

import pytest
from opentelemetry.instrumentation.gnap import GNAPInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


class FakeBoard:
    def __init__(self):
        self.agent_id = "agent-a"

    def create_task(self, task):
        return {"id": task["id"]}

    def claim_task(self, task_id):
        return {"id": task_id}

    def complete_task(self, task_id, result):
        return result


class AsyncFakeBoard:
    async def claim_task(self, task_id):
        raise asyncio.CancelledError


def test_gnap_task_lifecycle_creates_spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = GNAPInstrumentor()
    instrumentor._instrument(tracer_provider=provider, board_class=FakeBoard)
    try:
        board = FakeBoard()
        board.create_task({"id": "FA-1"})
        board.claim_task("FA-1")
        board.complete_task("FA-1", "done")
    finally:
        instrumentor._uninstrument()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "gnap.task.create",
        "gnap.task.claim",
        "gnap.task.complete",
    ]
    assert spans[0].attributes["gnap.task.id"] == "FA-1"
    assert spans[1].attributes["gnap.task.id"] == "FA-1"
    assert spans[2].attributes["gnap.task.id"] == "FA-1"
    assert spans[0].attributes["gnap.agent.id"] == "agent-a"
    assert spans[2].attributes["gnap.operation.success"] is True


@pytest.mark.parametrize(("result", "expected_size"), [("done", 4), (b"done", 4)])
def test_gnap_records_text_and_byte_result_size(result, expected_size):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = GNAPInstrumentor()
    instrumentor._instrument(tracer_provider=provider, board_class=FakeBoard)
    try:
        FakeBoard().complete_task("FA-1", result)
    finally:
        instrumentor._uninstrument()

    spans = exporter.get_finished_spans()
    assert spans[0].attributes["gnap.result.size"] == expected_size


@pytest.mark.asyncio
async def test_gnap_async_cancellation_ends_span_and_propagates():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = GNAPInstrumentor()
    instrumentor._instrument(tracer_provider=provider, board_class=AsyncFakeBoard)
    try:
        with pytest.raises(asyncio.CancelledError):
            await AsyncFakeBoard().claim_task("FA-1")
    finally:
        instrumentor._uninstrument()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["gnap.task.id"] == "FA-1"
