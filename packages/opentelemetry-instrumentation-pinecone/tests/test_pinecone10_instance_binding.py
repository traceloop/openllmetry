"""Pinecone 10 instance-binding regression tests (openllmetry #4716).

On pinecone 10.x, ``Index.__init__`` rebinds ``query``/``upsert`` on the
instance (``install_async_req_support``), so the class-level wrapping the
instrumentor applies is shadowed. These tests pin the two behaviours that
must hold on pinecone >= 10:

* instrumenting works at all (the version gate admits 10.x), and an
  ``Index`` created after instrumentation emits exactly one span per
  call, including on the error path (no network, unreachable host);
* the per-instance wrapping helper itself traces an instance-bound
  method exactly once.

Skipped entirely on pinecone < 10, where the instance binding does not
exist.
"""

import pinecone
import pytest

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.instrumentation.pinecone import _wrap_instance_methods

pytestmark = pytest.mark.skipif(
    tuple(int(p) for p in pinecone.__version__.split(".")[:2]) < (10, 0),
    reason="pinecone 10 instance binding only",
)


def test_query_upsert_delete_emit_exactly_one_span_each(traces_exporter):
    """Instrument-first order: one span per call, error path included.

    Relies on the session fixture (conftest) having instrumented already,
    as in the other tests in this package.
    """
    index = pinecone.Index(api_key="dummy", host="http://127.0.0.1:9")
    for method, kwargs in (
        ("query", dict(vector=[0.1, 0.2], top_k=1)),
        ("upsert", dict(vectors=[{"id": "v1", "values": [0.1, 0.2]}])),
        ("delete", dict(ids=["v1"])),
    ):
        with pytest.raises(Exception):
            getattr(index, method)(**kwargs)

    spans = traces_exporter.get_finished_spans()
    assert [s.name for s in spans] == [
        "pinecone.query",
        "pinecone.upsert",
        "pinecone.delete",
    ]
    for span in spans:
        assert span.status.status_code.name == "ERROR"
        assert span.attributes.get("server.address") == "http://127.0.0.1:9"


def test_wrap_instance_methods_traces_once():
    """The per-instance path itself: one span, no double counting."""

    class FakeConfig:
        host = "http://127.0.0.1:9"

    class FakeIndex:
        def __init__(self):
            self._config = FakeConfig()
            # Mimic pinecone 10: query/upsert live in the instance dict,
            # shadowing any class attribute.
            self.query = self._query
            self.upsert = self._upsert

        def _query(self, **kwargs):
            raise ConnectionError("unreachable host")

        def _upsert(self, **kwargs):
            raise ConnectionError("unreachable host")

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = trace.get_tracer(__name__, "test", provider)

    index = FakeIndex()
    _wrap_instance_methods(index, tracer, None, None, None, None)

    with pytest.raises(ConnectionError):
        index.query(vector=[0.1, 0.2], top_k=1)
    with pytest.raises(ConnectionError):
        index.upsert(vectors=[{"id": "v1", "values": [0.1, 0.2]}])

    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["pinecone.query", "pinecone.upsert"]
    for span in spans:
        assert span.status.status_code.name == "ERROR"
        assert span.attributes.get("server.address") == "http://127.0.0.1:9"
