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
    assert spans[0].attributes["gnap.agent.id"] == "agent-a"
    assert spans[2].attributes["gnap.operation.success"] is True
