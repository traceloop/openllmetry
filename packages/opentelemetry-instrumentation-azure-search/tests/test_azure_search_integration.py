"""Integration tests for Azure AI Search instrumentation using VCR cassettes.

Each test verifies that a real Azure SDK call (replayed from a recorded cassette)
produces a span with the correct name, kind, status, vendor, request attributes,
response attributes, and content capture attributes.
"""

import json
import os

import pytest
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import IndexDocumentsBatch, SearchClient
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    AnalyzeTextOptions,
    InputFieldMappingEntry,
    LanguageDetectionSkill,
    OutputFieldMappingEntry,
    SearchFieldDataType,
    SearchIndex,
    SearchIndexer,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SearchIndexerSkillset,
    SearchSuggester,
    SearchableField,
    SimpleField,
    SynonymMap,
)
from opentelemetry.semconv_ai import EventAttributes, SpanAttributes
from opentelemetry.trace import SpanKind, StatusCode

INTEGRATION_TEST_INDEX = "otel-integration-test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_only_span(exporter, span_name):
    """Return the single span matching span_name, or fail with a clear message."""
    spans = exporter.get_finished_spans()
    matching = [s for s in spans if s.name == span_name]
    assert len(matching) == 1, (
        f"Expected exactly 1 '{span_name}' span, "
        f"got {len(matching)} out of {len(spans)} total spans: "
        f"{[s.name for s in spans]}"
    )
    return matching[0]


def _assert_base_span(span, expected_name, index_name=None):
    """Verify the common properties every instrumented span must have."""
    assert span.name == expected_name
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.OK
    assert span.attributes[SpanAttributes.VECTOR_DB_VENDOR] == "Azure AI Search"
    if index_name is not None:
        assert span.attributes[SpanAttributes.AZURE_SEARCH_INDEX_NAME] == index_name


def _span_attrs(span):
    """Return span attributes as a plain dict for easier key/value access."""
    return dict(span.attributes)


def _assert_no_content_attributes(span):
    """Assert that zero content-capture attributes are present on a span."""
    content_prefixes = (
        EventAttributes.DB_QUERY_RESULT_DOCUMENT.value,
        EventAttributes.DB_SEARCH_RESULT_ENTITY.value,
        EventAttributes.DB_SEARCH_EMBEDDINGS_VECTOR.value,
        EventAttributes.DB_QUERY_RESULT_METADATA.value,
        EventAttributes.DB_QUERY_RESULT_ID.value,
    )
    attrs = _span_attrs(span)
    content_keys = [
        k for k in attrs
        if any(k.startswith(p + ".") for p in content_prefixes)
    ]
    assert content_keys == [], (
        f"Found content attributes with content tracing disabled: {content_keys}"
    )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_index_client():
    return SearchIndexClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]),
    )


def _make_search_client(index_name=INTEGRATION_TEST_INDEX):
    return SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=index_name,
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]),
    )


def _make_indexer_client():
    return SearchIndexerClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]),
    )


def _is_playback_mode():
    return os.environ.get("AZURE_SEARCH_ADMIN_KEY") == "test-api-key"


def _setup_index(index_client, fields, suggesters=None):
    """Create the integration-test index (idempotent). Skips in playback mode."""
    if _is_playback_mode():
        return
    try:
        index_client.delete_index(INTEGRATION_TEST_INDEX)
    except Exception:
        pass
    index = SearchIndex(
        name=INTEGRATION_TEST_INDEX,
        fields=fields,
        suggesters=suggesters or [],
    )
    index_client.create_index(index)


def _teardown_index(index_client):
    """Delete the integration-test index. Skips in playback mode."""
    if _is_playback_mode():
        return
    try:
        index_client.delete_index(INTEGRATION_TEST_INDEX)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# TestSearchClientIntegration
# ---------------------------------------------------------------------------

class TestSearchClientIntegration:
    """Integration tests for SearchClient operations (search, document CRUD, autocomplete, suggest)."""

    @pytest.fixture(scope="class")
    def index_client_setup(self):
        return _make_index_client()

    @pytest.fixture(scope="class", autouse=True)
    def setup_test_index(self, index_client_setup):
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="name", type=SearchFieldDataType.String),
            SearchableField(name="description", type=SearchFieldDataType.String),
            SimpleField(name="rating", type=SearchFieldDataType.Double, filterable=True),
        ]
        suggesters = [SearchSuggester(name="sg", source_fields=["name"])]
        _setup_index(index_client_setup, fields, suggesters)
        yield
        _teardown_index(index_client_setup)

    @pytest.fixture
    def search_client(self):
        return _make_search_client()

    @pytest.fixture
    def index_client(self):
        return _make_index_client()

    # -- Search operations --

    @pytest.mark.vcr
    def test_search(self, exporter, search_client):
        """Search with text, top, and filter captures all query parameters."""
        list(search_client.search(search_text="hotel", top=5, filter="rating ge 3"))

        span = _get_only_span(exporter, "azure_search.search")
        _assert_base_span(span, "azure_search.search", INTEGRATION_TEST_INDEX)
        assert span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_TEXT] == "hotel"
        assert span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_TOP] == 5
        assert span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_FILTER] == "rating ge 3"

    @pytest.mark.vcr
    def test_search_with_skip(self, exporter, search_client):
        """Search with skip parameter captures pagination offset."""
        list(search_client.search(search_text="*", top=10, skip=5))

        span = _get_only_span(exporter, "azure_search.search")
        _assert_base_span(span, "azure_search.search", INTEGRATION_TEST_INDEX)
        assert span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_SKIP] == 5

    # -- Document retrieval --

    @pytest.mark.vcr
    def test_get_document(self, exporter, search_client):
        """get_document captures the document key and the full document as a content attribute."""
        # Cassette includes a prior upload of doc-1, then the GET
        search_client.upload_documents([
            {"id": "doc-1", "name": "Test", "description": "Test", "rating": 4.0},
        ])
        exporter.clear()

        search_client.get_document(key="doc-1")

        span = _get_only_span(exporter, "azure_search.get_document")
        _assert_base_span(span, "azure_search.get_document", INTEGRATION_TEST_INDEX)
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_KEY] == "doc-1"

        # Verify the full document is captured as a content attribute
        attrs = _span_attrs(span)
        raw = attrs[EventAttributes.DB_QUERY_RESULT_DOCUMENT.value]
        captured_doc = json.loads(raw)
        assert captured_doc["id"] == "doc-1"
        assert captured_doc["name"] == "Test"
        assert captured_doc["rating"] == 4.0

    @pytest.mark.vcr
    def test_get_document_count(self, exporter, search_client):
        """get_document_count captures the count as a response attribute."""
        count = search_client.get_document_count()

        span = _get_only_span(exporter, "azure_search.get_document_count")
        _assert_base_span(span, "azure_search.get_document_count", INTEGRATION_TEST_INDEX)
        # The cassette returns "1" — verify the response attribute matches
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == count

    # -- Document write operations --

    @pytest.mark.vcr
    def test_upload_documents(self, exporter, search_client):
        """upload_documents captures doc count, succeeded/failed counts, and content."""
        documents = [
            {"id": "test-1", "name": "Test Hotel 1", "description": "A test hotel", "rating": 4.0},
            {"id": "test-2", "name": "Test Hotel 2", "description": "Another test hotel", "rating": 3.5},
        ]
        search_client.upload_documents(documents=documents)

        span = _get_only_span(exporter, "azure_search.upload_documents")
        _assert_base_span(span, "azure_search.upload_documents", INTEGRATION_TEST_INDEX)

        # Request attributes
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 2

        # Response attributes — cassette shows both docs succeed with 201
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT] == 2
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_FAILED_COUNT] == 0

        # Request content — each input document captured
        attrs = _span_attrs(span)
        doc_0 = json.loads(attrs[f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.0"])
        doc_1 = json.loads(attrs[f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.1"])
        assert doc_0["id"] == "test-1"
        assert doc_1["id"] == "test-2"

        # Response content — indexing result metadata captured
        meta_0 = json.loads(attrs[f"{EventAttributes.DB_QUERY_RESULT_METADATA.value}.0"])
        assert meta_0["succeeded"] is True
        assert meta_0["status_code"] == 201

    @pytest.mark.vcr
    def test_merge_documents(self, exporter, search_client):
        """merge_documents captures doc count and succeeded/failed counts."""
        # Cassette uploads merge-1 first, then merges it
        search_client.upload_documents([
            {"id": "merge-1", "name": "Merge Test", "description": "Test", "rating": 3.0},
        ])
        exporter.clear()

        search_client.merge_documents(documents=[{"id": "merge-1", "rating": 4.8}])

        span = _get_only_span(exporter, "azure_search.merge_documents")
        _assert_base_span(span, "azure_search.merge_documents", INTEGRATION_TEST_INDEX)
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 1
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT] == 1
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_FAILED_COUNT] == 0

    @pytest.mark.vcr
    def test_delete_documents(self, exporter, search_client):
        """delete_documents captures doc count and succeeded/failed counts."""
        search_client.delete_documents(documents=[{"id": "test-1"}, {"id": "test-2"}])

        span = _get_only_span(exporter, "azure_search.delete_documents")
        _assert_base_span(span, "azure_search.delete_documents", INTEGRATION_TEST_INDEX)
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 2
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT] == 2
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_FAILED_COUNT] == 0

    @pytest.mark.vcr
    def test_merge_or_upload_documents(self, exporter, search_client):
        """merge_or_upload_documents captures doc count and succeeded/failed counts."""
        search_client.merge_or_upload_documents(
            documents=[{"id": "upsert-1", "name": "Upsert Hotel", "description": "A test upsert", "rating": 4.2}],
        )

        span = _get_only_span(exporter, "azure_search.merge_or_upload_documents")
        _assert_base_span(span, "azure_search.merge_or_upload_documents", INTEGRATION_TEST_INDEX)
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 1
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT] == 1
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_FAILED_COUNT] == 0

    @pytest.mark.vcr
    def test_index_documents(self, exporter, search_client):
        """index_documents (batch API) captures batch size and succeeded/failed counts."""
        batch = IndexDocumentsBatch()
        batch.add_upload_actions([
            {"id": "batch-1", "name": "Batch Hotel", "description": "A batch test", "rating": 3.9},
        ])
        search_client.index_documents(batch=batch)

        span = _get_only_span(exporter, "azure_search.index_documents")
        _assert_base_span(span, "azure_search.index_documents", INTEGRATION_TEST_INDEX)
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 1
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT] == 1
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_FAILED_COUNT] == 0

    # -- Autocomplete & Suggest --

    @pytest.mark.vcr
    def test_autocomplete(self, exporter, search_client):
        """autocomplete captures search text, suggester, results count, and suggestion content."""
        # Cassette uploads auto-1 first, then autocompletes
        search_client.upload_documents([
            {"id": "auto-1", "name": "Luxury Hotel", "description": "A luxury hotel", "rating": 5.0},
        ])
        exporter.clear()

        list(search_client.autocomplete(search_text="lux", suggester_name="sg"))

        span = _get_only_span(exporter, "azure_search.autocomplete")
        _assert_base_span(span, "azure_search.autocomplete", INTEGRATION_TEST_INDEX)
        assert span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_TEXT] == "lux"
        assert span.attributes[SpanAttributes.AZURE_SEARCH_SUGGESTER_NAME] == "sg"

        # Cassette returns 1 result: {"text":"luxury","queryPlusText":"luxury"}
        assert span.attributes[SpanAttributes.AZURE_SEARCH_AUTOCOMPLETE_RESULTS_COUNT] == 1

        # Verify content capture — entity.0 should contain the suggestion
        attrs = _span_attrs(span)
        entity_0 = json.loads(attrs[f"{EventAttributes.DB_SEARCH_RESULT_ENTITY.value}.0"])
        assert entity_0["text"] == "luxury"
        assert entity_0["query_plus_text"] == "luxury"

    @pytest.mark.vcr
    def test_suggest(self, exporter, search_client):
        """suggest captures search text, suggester, results count, and suggestion content."""
        # Cassette uploads sug-1 first, then suggests
        search_client.upload_documents([
            {"id": "sug-1", "name": "Hot Springs Resort", "description": "A hot springs resort", "rating": 4.5},
        ])
        exporter.clear()

        results = list(search_client.suggest(search_text="hot", suggester_name="sg"))

        span = _get_only_span(exporter, "azure_search.suggest")
        _assert_base_span(span, "azure_search.suggest", INTEGRATION_TEST_INDEX)
        assert span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_TEXT] == "hot"
        assert span.attributes[SpanAttributes.AZURE_SEARCH_SUGGESTER_NAME] == "sg"

        # Cassette returns 4 suggestions
        assert span.attributes[SpanAttributes.AZURE_SEARCH_SUGGEST_RESULTS_COUNT] == len(results)
        assert span.attributes[SpanAttributes.AZURE_SEARCH_SUGGEST_RESULTS_COUNT] >= 1

        # Verify content capture — at least entity.0 is present and parseable
        attrs = _span_attrs(span)
        entity_0_raw = attrs[f"{EventAttributes.DB_SEARCH_RESULT_ENTITY.value}.0"]
        entity_0 = json.loads(entity_0_raw)
        assert "@search.text" in entity_0 or "id" in entity_0

    # -- Content toggle --

    @pytest.mark.vcr
    def test_content_disabled_no_content_attributes(self, exporter, search_client, monkeypatch):
        """With TRACELOOP_TRACE_CONTENT=false, get_document still creates a span but omits content."""
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

        search_client.get_document(key="1")

        span = _get_only_span(exporter, "azure_search.get_document")
        _assert_base_span(span, "azure_search.get_document", INTEGRATION_TEST_INDEX)

        # The document key is always captured (metadata, not content)
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_KEY] == "1"

        # But the document body must NOT be captured
        _assert_no_content_attributes(span)


# ---------------------------------------------------------------------------
# TestSearchIndexClientIntegration
# ---------------------------------------------------------------------------

class TestSearchIndexClientIntegration:
    """Integration tests for SearchIndexClient operations (index CRUD, stats, analyze)."""

    @pytest.fixture(scope="class")
    def index_client_setup(self):
        return _make_index_client()

    @pytest.fixture(scope="class", autouse=True)
    def setup_test_index(self, index_client_setup):
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="name", type=SearchFieldDataType.String),
            SearchableField(name="description", type=SearchFieldDataType.String),
            SimpleField(name="rating", type=SearchFieldDataType.Double, filterable=True),
        ]
        suggesters = [SearchSuggester(name="sg", source_fields=["name"])]
        _setup_index(index_client_setup, fields, suggesters)
        yield
        _teardown_index(index_client_setup)

    @pytest.fixture
    def index_client(self):
        return _make_index_client()

    # -- Index retrieval --

    @pytest.mark.vcr
    def test_list_indexes(self, exporter, index_client):
        """list_indexes produces a span with vendor attribute."""
        list(index_client.list_indexes())

        span = _get_only_span(exporter, "azure_search.list_indexes")
        _assert_base_span(span, "azure_search.list_indexes")

    @pytest.mark.vcr
    def test_list_index_names(self, exporter, index_client):
        """list_index_names produces a span with vendor attribute."""
        list(index_client.list_index_names())

        span = _get_only_span(exporter, "azure_search.list_index_names")
        _assert_base_span(span, "azure_search.list_index_names")

    @pytest.mark.vcr
    def test_get_index(self, exporter, index_client):
        """get_index captures the index name."""
        index_client.get_index(INTEGRATION_TEST_INDEX)

        span = _get_only_span(exporter, "azure_search.get_index")
        _assert_base_span(span, "azure_search.get_index", INTEGRATION_TEST_INDEX)

    @pytest.mark.vcr
    def test_get_index_statistics(self, exporter, index_client):
        """get_index_statistics captures the index name."""
        index_client.get_index_statistics(INTEGRATION_TEST_INDEX)

        span = _get_only_span(exporter, "azure_search.get_index_statistics")
        _assert_base_span(span, "azure_search.get_index_statistics", INTEGRATION_TEST_INDEX)

    # -- Index CRUD --

    @pytest.mark.vcr
    def test_create_index(self, exporter, index_client):
        """create_index captures the new index name."""
        test_index_name = "test-create-index"

        # Cassette includes a prior delete (404) then create (201)
        try:
            index_client.delete_index(test_index_name)
        except Exception:
            pass
        exporter.clear()

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
        ]
        index_client.create_index(SearchIndex(name=test_index_name, fields=fields))

        try:
            span = _get_only_span(exporter, "azure_search.create_index")
            _assert_base_span(span, "azure_search.create_index", test_index_name)
        finally:
            try:
                index_client.delete_index(test_index_name)
            except Exception:
                pass

    @pytest.mark.vcr
    def test_create_or_update_index(self, exporter, index_client):
        """create_or_update_index captures the index name."""
        test_index_name = "test-upsert-index"

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
        ]
        index_client.create_or_update_index(SearchIndex(name=test_index_name, fields=fields))

        try:
            span = _get_only_span(exporter, "azure_search.create_or_update_index")
            _assert_base_span(span, "azure_search.create_or_update_index", test_index_name)
        finally:
            try:
                index_client.delete_index(test_index_name)
            except Exception:
                pass

    @pytest.mark.vcr
    def test_delete_index(self, exporter, index_client):
        """delete_index captures the deleted index name."""
        test_index_name = "test-delete-index"

        # Cassette includes a prior create (201) then the delete (204)
        fields = [SimpleField(name="id", type=SearchFieldDataType.String, key=True)]
        try:
            index_client.create_index(SearchIndex(name=test_index_name, fields=fields))
        except Exception:
            pass
        exporter.clear()

        index_client.delete_index(test_index_name)

        span = _get_only_span(exporter, "azure_search.delete_index")
        _assert_base_span(span, "azure_search.delete_index", test_index_name)

    # -- Text analysis --

    @pytest.mark.vcr
    def test_analyze_text(self, exporter, index_client):
        """analyze_text captures the index name and analyzer name."""
        index_client.analyze_text(
            index_name=INTEGRATION_TEST_INDEX,
            analyze_request=AnalyzeTextOptions(
                text="The quick brown fox",
                analyzer_name="standard.lucene",
            ),
        )

        span = _get_only_span(exporter, "azure_search.analyze_text")
        _assert_base_span(span, "azure_search.analyze_text", INTEGRATION_TEST_INDEX)
        assert span.attributes[SpanAttributes.AZURE_SEARCH_ANALYZER_NAME] == "standard.lucene"

    # -- Service statistics --

    @pytest.mark.vcr
    def test_get_service_statistics(self, exporter, index_client):
        """get_service_statistics captures document and index counts from the response."""
        index_client.get_service_statistics()

        span = _get_only_span(exporter, "azure_search.get_service_statistics")
        _assert_base_span(span, "azure_search.get_service_statistics")

        # Cassette shows indexesCount=2, verify response attributes are integers
        doc_count = span.attributes[SpanAttributes.AZURE_SEARCH_SERVICE_DOCUMENT_COUNT]
        idx_count = span.attributes[SpanAttributes.AZURE_SEARCH_SERVICE_INDEX_COUNT]
        assert isinstance(doc_count, int)
        assert isinstance(idx_count, int)
        assert idx_count >= 1  # At least the integration test index exists


# ---------------------------------------------------------------------------
# TestSynonymMapIntegration
# ---------------------------------------------------------------------------

class TestSynonymMapIntegration:
    """Integration tests for synonym map CRUD operations."""

    @pytest.fixture
    def index_client(self):
        return _make_index_client()

    # -- CRUD --

    @pytest.mark.vcr
    def test_create_synonym_map(self, exporter, index_client):
        """create_synonym_map captures name and synonym count."""
        sm_name = "otel-test-synonyms"

        try:
            index_client.delete_synonym_map(sm_name)
        except Exception:
            pass
        exporter.clear()

        index_client.create_synonym_map(
            SynonymMap(name=sm_name, synonyms=["hotel,motel", "cozy,comfortable,warm"]),
        )

        try:
            span = _get_only_span(exporter, "azure_search.create_synonym_map")
            _assert_base_span(span, "azure_search.create_synonym_map")
            assert span.attributes[SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_NAME] == sm_name
            assert span.attributes[SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_SYNONYMS_COUNT] == 2
        finally:
            try:
                index_client.delete_synonym_map(sm_name)
            except Exception:
                pass

    @pytest.mark.vcr
    def test_create_or_update_synonym_map(self, exporter, index_client):
        """create_or_update_synonym_map captures name and synonym count."""
        sm_name = "otel-test-upsert-sm"

        index_client.create_or_update_synonym_map(
            SynonymMap(name=sm_name, synonyms=["fast,quick", "slow,sluggish"]),
        )

        try:
            span = _get_only_span(exporter, "azure_search.create_or_update_synonym_map")
            _assert_base_span(span, "azure_search.create_or_update_synonym_map")
            assert span.attributes[SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_NAME] == sm_name
            assert span.attributes[SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_SYNONYMS_COUNT] == 2
        finally:
            try:
                index_client.delete_synonym_map(sm_name)
            except Exception:
                pass

    @pytest.mark.vcr
    def test_get_synonym_map(self, exporter, index_client):
        """get_synonym_map captures the synonym map name."""
        sm_name = "otel-test-get-sm"

        try:
            index_client.create_synonym_map(SynonymMap(name=sm_name, synonyms=["big,large"]))
        except Exception:
            pass
        exporter.clear()

        index_client.get_synonym_map(sm_name)

        try:
            span = _get_only_span(exporter, "azure_search.get_synonym_map")
            _assert_base_span(span, "azure_search.get_synonym_map")
            assert span.attributes[SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_NAME] == sm_name
        finally:
            try:
                index_client.delete_synonym_map(sm_name)
            except Exception:
                pass

    @pytest.mark.vcr
    def test_get_synonym_maps(self, exporter, index_client):
        """get_synonym_maps produces a span with vendor attribute."""
        sm_name = "otel-test-list-sms"

        try:
            index_client.create_synonym_map(
                SynonymMap(name=sm_name, synonyms=["hello,hi", "goodbye,bye"]),
            )
        except Exception:
            pass
        exporter.clear()

        index_client.get_synonym_maps()

        try:
            span = _get_only_span(exporter, "azure_search.get_synonym_maps")
            _assert_base_span(span, "azure_search.get_synonym_maps")
        finally:
            try:
                index_client.delete_synonym_map(sm_name)
            except Exception:
                pass

    @pytest.mark.vcr
    def test_get_synonym_map_names(self, exporter, index_client):
        """get_synonym_map_names produces a span with vendor attribute."""
        sm_name = "otel-test-list-sm-names"

        try:
            index_client.create_synonym_map(SynonymMap(name=sm_name, synonyms=["warm,cozy"]))
        except Exception:
            pass
        exporter.clear()

        index_client.get_synonym_map_names()

        try:
            span = _get_only_span(exporter, "azure_search.get_synonym_map_names")
            _assert_base_span(span, "azure_search.get_synonym_map_names")
        finally:
            try:
                index_client.delete_synonym_map(sm_name)
            except Exception:
                pass

    @pytest.mark.vcr
    def test_delete_synonym_map(self, exporter, index_client):
        """delete_synonym_map captures the synonym map name."""
        sm_name = "otel-test-delete-sm"

        try:
            index_client.create_synonym_map(SynonymMap(name=sm_name, synonyms=["old,ancient"]))
        except Exception:
            pass
        exporter.clear()

        index_client.delete_synonym_map(sm_name)

        span = _get_only_span(exporter, "azure_search.delete_synonym_map")
        _assert_base_span(span, "azure_search.delete_synonym_map")
        assert span.attributes[SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_NAME] == sm_name


# ---------------------------------------------------------------------------
# TestSearchIndexerClientIntegration
# ---------------------------------------------------------------------------

class TestSearchIndexerClientIntegration:
    """Integration tests for SearchIndexerClient name-only listing methods.

    NOTE: The data source and indexer setup uses placeholder credentials that fail
    with 400 during recording. The cassettes still capture the listing calls with
    empty results, which is sufficient to verify span creation. Re-record with
    valid Azure Storage credentials to test non-empty listings.
    """

    @pytest.fixture(scope="class")
    def index_client_setup(self):
        return _make_index_client()

    @pytest.fixture(scope="class", autouse=True)
    def setup_test_index(self, index_client_setup):
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="name", type=SearchFieldDataType.String),
        ]
        _setup_index(index_client_setup, fields)
        yield
        _teardown_index(index_client_setup)

    @pytest.fixture
    def indexer_client(self):
        return _make_indexer_client()

    @pytest.mark.vcr
    def test_get_indexer_names(self, exporter, indexer_client):
        """get_indexer_names produces a span even with failed setup (placeholder credentials)."""
        ds_name = "otel-test-indexer-ds"
        indexer_name = "otel-test-indexer-names"

        ds_connection = SearchIndexerDataSourceConnection(
            name=ds_name,
            type="azureblob",
            connection_string=os.environ.get(
                "AZURE_STORAGE_CONNECTION_STRING",
                "DefaultEndpointsProtocol=https;AccountName=placeholder;"
                "AccountKey=placeholder;EndpointSuffix=core.windows.net",
            ),
            container=SearchIndexerDataContainer(name="placeholder-container"),
        )
        indexer = SearchIndexer(
            name=indexer_name,
            data_source_name=ds_name,
            target_index_name=INTEGRATION_TEST_INDEX,
            is_disabled=True,
        )
        try:
            indexer_client.create_data_source_connection(ds_connection)
            indexer_client.create_indexer(indexer)
        except Exception:
            pass  # Expected to fail with placeholder credentials
        exporter.clear()

        try:
            list(indexer_client.get_indexer_names())

            span = _get_only_span(exporter, "azure_search.get_indexer_names")
            _assert_base_span(span, "azure_search.get_indexer_names")
        finally:
            for name, delete_fn in [
                (indexer_name, indexer_client.delete_indexer),
                (ds_name, indexer_client.delete_data_source_connection),
            ]:
                try:
                    delete_fn(name)
                except Exception:
                    pass

    @pytest.mark.vcr
    def test_get_data_source_connection_names(self, exporter, indexer_client):
        """get_data_source_connection_names produces a span."""
        ds_name = "otel-test-ds-names"

        ds_connection = SearchIndexerDataSourceConnection(
            name=ds_name,
            type="azureblob",
            connection_string=os.environ.get(
                "AZURE_STORAGE_CONNECTION_STRING",
                "DefaultEndpointsProtocol=https;AccountName=placeholder;"
                "AccountKey=placeholder;EndpointSuffix=core.windows.net",
            ),
            container=SearchIndexerDataContainer(name="placeholder-container"),
        )
        try:
            indexer_client.create_data_source_connection(ds_connection)
        except Exception:
            pass  # Expected to fail with placeholder credentials
        exporter.clear()

        try:
            list(indexer_client.get_data_source_connection_names())

            span = _get_only_span(exporter, "azure_search.get_data_source_connection_names")
            _assert_base_span(span, "azure_search.get_data_source_connection_names")
        finally:
            try:
                indexer_client.delete_data_source_connection(ds_name)
            except Exception:
                pass

    @pytest.mark.vcr
    def test_get_skillset_names(self, exporter, indexer_client):
        """get_skillset_names produces a span (skillset creation succeeds without storage)."""
        skillset_name = "otel-test-skillset-names"

        skillset = SearchIndexerSkillset(
            name=skillset_name,
            skills=[
                LanguageDetectionSkill(
                    name="language-detection",
                    inputs=[InputFieldMappingEntry(name="text", source="/document/content")],
                    outputs=[OutputFieldMappingEntry(name="languageCode", target_name="languageCode")],
                ),
            ],
            description="Test skillset for OTel integration tests",
        )
        try:
            indexer_client.create_skillset(skillset)
        except Exception:
            pass
        exporter.clear()

        try:
            list(indexer_client.get_skillset_names())

            span = _get_only_span(exporter, "azure_search.get_skillset_names")
            _assert_base_span(span, "azure_search.get_skillset_names")
        finally:
            try:
                indexer_client.delete_skillset(skillset_name)
            except Exception:
                pass
