from unittest.mock import patch

import pytest
from opentelemetry.trace.status import StatusCode
from pinecone import Pinecone


def test_pinecone_query_error_sets_span_status_and_error_type(traces_exporter):
    with patch(
        "urllib3.connectionpool.HTTPSConnectionPool.urlopen",
        side_effect=Exception("Connection error"),
    ):
        with pytest.raises(Exception):
            pc = Pinecone(api_key="bad-key")
            index = pc.Index(host="https://test-index.pinecone.io")
            index.query(vector=[0.1] * 1536, top_k=1)

    spans = traces_exporter.get_finished_spans()
    query_spans = [
        s for s in spans
        if s.name == "pinecone.query" and s.status.status_code == StatusCode.ERROR
    ]
    assert len(query_spans) == 1
    span = query_spans[0]

    assert span.attributes.get("error.type") == "Exception"
    events = [e for e in span.events if e.name == "exception"]
    assert len(events) == 1
    assert "Connection error" in events[0].attributes["exception.message"]
