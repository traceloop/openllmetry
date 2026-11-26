import pytest
from unittest.mock import patch, MagicMock
from azure.search.documents import SearchClient


def _make_client():
    client = SearchClient.__new__(SearchClient)
    client._index_name = "test-index"
    client._endpoint = "https://test.search.windows.net"
    return client


def test_search_creates_span(exporter):
    client = _make_client()
    with patch("azure.core.pipeline.Pipeline.run", return_value=MagicMock(http_response=MagicMock(status_code=200, text=lambda encoding=None: '{"value": []}', headers={}))):
        try:
            client.search("hello world", top=5)
        except Exception:
            pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "azure_search.search"


def test_search_span_attributes(exporter):
    client = _make_client()
    client._index_name = "my-index"
    with patch("azure.core.pipeline.Pipeline.run", return_value=MagicMock(http_response=MagicMock(status_code=200, text=lambda encoding=None: '{"value": []}', headers={}))):
        try:
            client.search("test query", top=10)
        except Exception:
            pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get("db.system") == "azure_search"


def test_search_error_captured(exporter):
    client = _make_client()
    with patch("azure.core.pipeline.Pipeline.run", side_effect=Exception("Service unavailable")):
        try:
            results = client.search("query")
            list(results)
        except Exception:
            pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "azure_search.search"