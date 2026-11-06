import pytest
from unittest.mock import patch, MagicMock
from azure.search.documents import SearchClient


def _make_client():
    client = SearchClient.__new__(SearchClient)
    client._index_name = "test-index"
    client._endpoint = "https://test.search.windows.net"
    return client


def _mock_pipeline_response():
    return MagicMock(
        http_response=MagicMock(
            status_code=200,
            text=lambda encoding=None: '{"value": []}',
            headers={}
        )
    )


def test_upload_documents_creates_span(exporter):
    client = _make_client()
    with patch("azure.core.pipeline.Pipeline.run", return_value=_mock_pipeline_response()):
        try:
            client.upload_documents(documents=[{"id": "1"}, {"id": "2"}])
        except Exception:
            pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "azure_search.upload_documents"


def test_merge_documents_creates_span(exporter):
    client = _make_client()
    with patch("azure.core.pipeline.Pipeline.run", return_value=_mock_pipeline_response()):
        try:
            client.merge_documents(documents=[{"id": "1", "title": "updated"}])
        except Exception:
            pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "azure_search.merge_documents"


def test_delete_documents_error_captured(exporter):
    client = _make_client()
    with patch("azure.search.documents._patch.SearchClient.index_documents", side_effect=RuntimeError("Auth failed")):
        with pytest.raises(RuntimeError):
            client.delete_documents(documents=[{"id": "1"}])

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    from opentelemetry.trace import StatusCode
    assert spans[0].status.status_code == StatusCode.ERROR