"""Unit tests for Azure AI Search instrumentation.

Uses real Azure Search SDK clients whose HTTP transport is replaced with a
fake, so every call goes through the instrumented SDK methods without any
network access. Spans are verified via the InMemorySpanExporter from
conftest.py.
"""

import json

import pytest
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ServiceRequestError
from azure.core.pipeline.transport import HttpResponse
from azure.core.utils import CaseInsensitiveDict
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import SearchIndexer
from opentelemetry.semconv_ai import SpanAttributes

ENDPOINT = "https://test.search.windows.net"

INDEX_BODY = json.dumps(
    {
        "name": "my-index",
        "fields": [{"name": "id", "type": "Edm.String", "key": True}],
    }
).encode()

INDEXING_RESULTS_BODY = json.dumps(
    {
        "value": [
            {"key": "1", "status": 200, "succeeded": True, "errorMessage": None},
            {"key": "2", "status": 200, "succeeded": True, "errorMessage": None},
        ]
    }
).encode()

INDEXER_BODY = json.dumps(
    {
        "name": "my-indexer",
        "description": "",
        "dataSourceName": "ds",
        "targetIndexName": "my-index",
    }
).encode()

SKILLSET_BODY = json.dumps({"name": "my-skillset", "skills": []}).encode()


class FakeResponse(HttpResponse):
    """Minimal in-memory HttpResponse for the azure-core pipeline."""

    def __init__(self, request, body=b"", status_code=200):
        super().__init__(request, None)
        self.status_code = status_code
        self.headers = CaseInsensitiveDict(
            {"content-type": "application/json; charset=utf-8"} if status_code != 204 else {}
        )
        self._body_bytes = body

    def body(self):
        return self._body_bytes

    def text(self, encoding=None):
        return self._body_bytes.decode(encoding or "utf-8")

    def json(self):
        return json.loads(self._body_bytes.decode("utf-8"))


class FakeTransport:
    """Replaces the real HTTP transport with canned responses."""

    def __init__(self, body=b"", status_code=200, exception=None):
        self._body = body
        self._status_code = status_code
        self._exception = exception

    def send(self, request, **kwargs):
        if self._exception is not None:
            raise self._exception
        return FakeResponse(request, self._body, self._status_code)


def _client(cls, body=b"", status_code=200, exception=None, **kwargs):
    return cls(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential("fake-key"),
        transport=FakeTransport(body=body, status_code=status_code, exception=exception),
        retry_total=0,
        **kwargs,
    )


def _make_search_client(body=b"", **kwargs):
    return _client(SearchClient, body=body, index_name="test-index", **kwargs)


def _make_index_client(body=b"", status_code=200, exception=None):
    return _client(SearchIndexClient, body=body, status_code=status_code, exception=exception)


def _make_indexer_client(body=b"", status_code=200, exception=None):
    return _client(SearchIndexerClient, body=body, status_code=status_code, exception=exception)


# ---------------------------------------------------------------------------
# SearchClient tests
# ---------------------------------------------------------------------------


def test_search_creates_span(exporter):
    client = _make_search_client(body=b'{"value": [], "count": 0}')

    client.search(search_text="hello world", top=5, filter="category eq 'docs'")

    spans = exporter.get_finished_spans()
    search_spans = [s for s in spans if s.name == "azure_search.search"]
    assert len(search_spans) == 1

    span = search_spans[0]
    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "azure_search"
    assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_TEXT) == "hello world"
    assert span.attributes.get(SpanAttributes.AZURE_SEARCH_TOP) == 5
    assert span.attributes.get(SpanAttributes.AZURE_SEARCH_FILTER) == "category eq 'docs'"
    assert span.attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "test-index"
    assert span.attributes.get("server.address") == "test.search.windows.net"


def test_get_document_creates_span(exporter):
    client = _make_search_client(body=json.dumps({"key": "1", "title": "Test"}).encode())

    client.get_document(key="1")

    spans = exporter.get_finished_spans()
    get_doc_spans = [s for s in spans if s.name == "azure_search.get_document"]
    assert len(get_doc_spans) == 1

    span = get_doc_spans[0]
    assert span.attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "test-index"


def test_autocomplete_creates_span(exporter):
    client = _make_search_client(body=b'{"value": []}')

    client.autocomplete(search_text="hel", suggester_name="sg")

    spans = exporter.get_finished_spans()
    ac_spans = [s for s in spans if s.name == "azure_search.autocomplete"]
    assert len(ac_spans) == 1

    span = ac_spans[0]
    assert span.attributes.get(SpanAttributes.AZURE_SEARCH_AUTOCOMPLETE_TEXT) == "hel"


def test_suggest_creates_span(exporter):
    client = _make_search_client(body=b'{"value": []}')

    client.suggest(search_text="hel", suggester_name="sg")

    spans = exporter.get_finished_spans()
    suggest_spans = [s for s in spans if s.name == "azure_search.suggest"]
    assert len(suggest_spans) == 1

    span = suggest_spans[0]
    assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SUGGEST_TEXT) == "hel"


def test_index_documents_creates_span(exporter):
    client = _make_search_client(body=INDEXING_RESULTS_BODY)

    client.upload_documents(documents=[{"id": "1"}, {"id": "2"}])

    spans = exporter.get_finished_spans()
    idx_spans = [s for s in spans if s.name == "azure_search.upload_documents"]
    assert len(idx_spans) == 1

    span = idx_spans[0]
    assert span.attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENTS_COUNT) == 2
    assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SUCCEEDED_COUNT) == 2


def test_delete_documents_creates_span(exporter):
    body = json.dumps(
        {
            "value": [
                {"key": "1", "status": 200, "succeeded": True, "errorMessage": None},
                {"key": "2", "status": 200, "succeeded": True, "errorMessage": None},
                {"key": "3", "status": 200, "succeeded": True, "errorMessage": None},
            ]
        }
    ).encode()
    client = _make_search_client(body=body)

    client.delete_documents(documents=[{"id": "1"}, {"id": "2"}, {"id": "3"}])

    spans = exporter.get_finished_spans()
    del_spans = [s for s in spans if s.name == "azure_search.delete_documents"]
    assert len(del_spans) == 1
    assert del_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENTS_COUNT) == 3


def test_get_document_count_creates_span(exporter):
    client = _make_search_client(body=b"42")

    client.get_document_count()

    spans = exporter.get_finished_spans()
    count_spans = [s for s in spans if s.name == "azure_search.get_document_count"]
    assert len(count_spans) == 1
    assert count_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENTS_COUNT) == 42


# ---------------------------------------------------------------------------
# SearchIndexClient tests
# ---------------------------------------------------------------------------


def test_create_index_creates_span(exporter):
    client = _make_index_client(body=INDEX_BODY, status_code=201)

    client.create_index(index={"name": "my-index", "fields": []})

    spans = exporter.get_finished_spans()
    create_spans = [s for s in spans if s.name == "azure_search.create_index"]
    assert len(create_spans) == 1
    assert create_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "my-index"


def test_delete_index_creates_span(exporter):
    client = _make_index_client(status_code=204)

    client.delete_index(index="my-index")

    spans = exporter.get_finished_spans()
    del_spans = [s for s in spans if s.name == "azure_search.delete_index"]
    assert len(del_spans) == 1
    assert del_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "my-index"


def test_get_index_creates_span(exporter):
    client = _make_index_client(body=INDEX_BODY)

    client.get_index(name="my-index")

    spans = exporter.get_finished_spans()
    get_spans = [s for s in spans if s.name == "azure_search.get_index"]
    assert len(get_spans) == 1
    assert get_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "my-index"


def test_get_index_statistics_creates_span(exporter):
    body = json.dumps({"documentCount": 1000, "storageSize": 1048576}).encode()
    client = _make_index_client(body=body)

    client.get_index_statistics(index_name="my-index")

    spans = exporter.get_finished_spans()
    stat_spans = [s for s in spans if s.name == "azure_search.get_index_statistics"]
    assert len(stat_spans) == 1
    assert stat_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_DOC_COUNT) == 1000
    assert stat_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_SIZE_BYTES) == 1048576


def test_get_service_statistics_creates_span(exporter):
    body = json.dumps({"counters": {"documentCount": {"usage": 5, "quota": 100}}}).encode()
    client = _make_index_client(body=body)

    client.get_service_statistics()

    spans = exporter.get_finished_spans()
    svc_spans = [s for s in spans if s.name == "azure_search.get_service_statistics"]
    assert len(svc_spans) == 1


# ---------------------------------------------------------------------------
# SearchIndexerClient tests
# ---------------------------------------------------------------------------


def test_create_indexer_creates_span(exporter):
    client = _make_indexer_client(body=INDEXER_BODY, status_code=201)

    indexer = SearchIndexer(name="my-indexer", data_source_name="ds", target_index_name="i")
    client.create_indexer(indexer=indexer)

    spans = exporter.get_finished_spans()
    idx_spans = [s for s in spans if s.name == "azure_search.create_indexer"]
    assert len(idx_spans) == 1
    assert idx_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEXER_NAME) == "my-indexer"


def test_create_or_update_indexer_records_indexer_name(exporter):
    client = _make_indexer_client(body=INDEXER_BODY)

    indexer = SearchIndexer(name="my-indexer", data_source_name="ds", target_index_name="i")
    client.create_or_update_indexer(indexer=indexer)

    spans = exporter.get_finished_spans()
    idx_spans = [s for s in spans if s.name == "azure_search.create_or_update_indexer"]
    assert len(idx_spans) == 1
    assert idx_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEXER_NAME) == "my-indexer"


def test_get_indexer_status_creates_span(exporter):
    client = _make_indexer_client(body=json.dumps({"status": "running"}).encode())

    client.get_indexer_status(name="my-indexer")

    spans = exporter.get_finished_spans()
    status_spans = [s for s in spans if s.name == "azure_search.get_indexer_status"]
    assert len(status_spans) == 1
    assert status_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEXER_STATUS) == "running"


def test_run_indexer_creates_span(exporter):
    client = _make_indexer_client(status_code=202)

    client.run_indexer(name="my-indexer")

    spans = exporter.get_finished_spans()
    run_spans = [s for s in spans if s.name == "azure_search.run_indexer"]
    assert len(run_spans) == 1
    assert run_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEXER_NAME) == "my-indexer"


def test_create_skillset_records_skillset_name(exporter):
    client = _make_indexer_client(body=SKILLSET_BODY, status_code=201)

    client.create_skillset(skillset={"name": "my-skillset", "skills": []})

    spans = exporter.get_finished_spans()
    skillset_spans = [s for s in spans if s.name == "azure_search.create_skillset"]
    assert len(skillset_spans) == 1
    assert skillset_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SKILLSET_NAME) == "my-skillset"


def test_get_skillset_records_skillset_name(exporter):
    client = _make_indexer_client(body=SKILLSET_BODY)

    client.get_skillset(name="my-skillset")

    spans = exporter.get_finished_spans()
    skillset_spans = [s for s in spans if s.name == "azure_search.get_skillset"]
    assert len(skillset_spans) == 1
    assert skillset_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SKILLSET_NAME) == "my-skillset"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_upload_exception_records_exception(exporter):
    client = _make_search_client(exception=ConnectionResetError("Connection refused"))

    with pytest.raises((ConnectionResetError, ServiceRequestError)):
        client.upload_documents(documents=[{"id": "1"}])

    spans = exporter.get_finished_spans()
    search_spans = [s for s in spans if s.name == "azure_search.upload_documents"]
    assert len(search_spans) == 1

    span = search_spans[0]
    assert span.status.status_code.name == "ERROR"
    assert len(span.events) >= 1
    assert span.events[0].name == "exception"


# ---------------------------------------------------------------------------
# Instrumentor API
# ---------------------------------------------------------------------------


def test_instrumentor_has_correct_interface():
    from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor

    instrumentor = AzureSearchInstrumentor()
    assert hasattr(instrumentor, "_instrument")
    assert hasattr(instrumentor, "_uninstrument")
    assert "azure-search-documents" in instrumentor.instrumentation_dependencies()[0]
