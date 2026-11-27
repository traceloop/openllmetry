"""Tests for pinecone 10 support (issue #4716).

The span tests run against a local fake Pinecone data-plane server, so they
need no API key and no cassette and run on every supported pinecone version.
The fake responses carry zero scores and no usage, so these tests do not add
data points to the session-wide score / usage metrics that
test_pinecone_retrieval checks.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from opentelemetry.instrumentation.pinecone import (
    PineconeInstrumentor,
    _set_response_attributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace.status import StatusCode
from pinecone import Pinecone

QUERY_RESPONSE = {
    "matches": [{"id": "doc-1", "score": 0.0}, {"id": "doc-2", "score": 0.0}],
    "namespace": "",
}
UPSERT_RESPONSE = {"upsertedCount": 1}


class _FakePineconeHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        body = QUERY_RESPONSE if self.path.endswith("/query") else UPSERT_RESPONSE
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def index():
    server = HTTPServer(("127.0.0.1", 0), _FakePineconeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Pinecone(api_key="test_api_key").Index(host=f"http://127.0.0.1:{server.server_port}")
    finally:
        server.shutdown()
        server.server_close()


def test_supported_versions_include_pinecone_10():
    deps = PineconeInstrumentor().instrumentation_dependencies()
    assert deps == ("pinecone >= 5.1.0, <11",)


def test_pinecone_query_span(traces_exporter, index):
    index.query(vector=[0.1, 0.2], top_k=2)

    spans = [s for s in traces_exporter.get_finished_spans() if s.name == "pinecone.query"]
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.OK
    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Pinecone"
    assert span.attributes.get(SpanAttributes.PINECONE_QUERY_TOP_K) == 2
    assert span.attributes.get(SpanAttributes.PINECONE_QUERY_RESULT_COUNT) == 2


def test_pinecone_upsert_span(traces_exporter, index):
    index.upsert(vectors=[("doc-1", [0.1, 0.2])])

    spans = [s for s in traces_exporter.get_finished_spans() if s.name == "pinecone.upsert"]
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.OK
    assert spans[0].attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Pinecone"


@pytest.mark.parametrize(
    "usage",
    [
        # pinecone < 10: usage is dict-like
        {"read_units": 5},
        # pinecone 10: usage is a struct exposing read_units / write_units attributes only
        SimpleNamespace(read_units=5, write_units=None),
    ],
    ids=["dict-usage", "struct-usage"],
)
def test_usage_units_recorded(usage):
    span = MagicMock()
    read_units_metric = MagicMock()
    write_units_metric = MagicMock()

    _set_response_attributes(span, read_units_metric, write_units_metric, {}, {"usage": usage})

    span.set_attribute.assert_any_call(SpanAttributes.PINECONE_USAGE_READ_UNITS, 5)
    span.set_attribute.assert_any_call(SpanAttributes.PINECONE_USAGE_WRITE_UNITS, 0)
    read_units_metric.add.assert_called_once_with(5, {})
    write_units_metric.add.assert_called_once_with(0, {})
