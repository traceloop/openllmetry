"""Integration tests for Azure AI Search instrumentation using VCR cassettes."""

import os
import pytest
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchFieldDataType,
    SearchableField,
    SimpleField,
    SearchSuggester,
    AnalyzeTextOptions,
    SynonymMap,
    SearchIndexer,
    SearchIndexerDataSourceConnection,
    SearchIndexerDataContainer,
    SearchIndexerSkillset,
    LanguageDetectionSkill,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
)
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents import IndexDocumentsBatch
from opentelemetry.semconv_ai import SpanAttributes, EventAttributes
from opentelemetry.trace import SpanKind


# Test index name for integration tests that need a specific schema
INTEGRATION_TEST_INDEX = "otel-integration-test"


class TestSearchClientIntegration:
    """Integration tests for SearchClient instrumentation."""

    @pytest.fixture(scope="class")
    def index_client_setup(self):
        """Create a SearchIndexClient for setting up test index."""
        return SearchIndexClient(
            endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]),
        )

    @pytest.fixture(scope="class", autouse=True)
    def setup_test_index(self, index_client_setup):
        """Set up the test index with required schema before all tests, tear down after."""
        # Skip setup/teardown in playback mode (when using test credentials)
        is_playback_mode = os.environ.get("AZURE_SEARCH_ADMIN_KEY") == "test-api-key"

        if not is_playback_mode:
            # Define index schema
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="name", type=SearchFieldDataType.String),
                SearchableField(name="description", type=SearchFieldDataType.String),
                SimpleField(name="rating", type=SearchFieldDataType.Double, filterable=True),
            ]
            suggesters = [SearchSuggester(name="sg", source_fields=["name"])]
            index = SearchIndex(name=INTEGRATION_TEST_INDEX, fields=fields, suggesters=suggesters)

            # Delete index if it exists, then create fresh
            try:
                index_client_setup.delete_index(INTEGRATION_TEST_INDEX)
            except Exception:
                pass  # Index may not exist

            # Create the index
            index_client_setup.create_index(index)

        yield

        if not is_playback_mode:
            # Teardown: Delete the index and all documents after all tests
            try:
                index_client_setup.delete_index(INTEGRATION_TEST_INDEX)
            except Exception:
                pass  # Clean up best effort

    @pytest.fixture
    def search_client(self, exporter):
        """Create a SearchClient for testing."""
        return SearchClient(
            endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
            index_name=INTEGRATION_TEST_INDEX,
            credential=AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]),
        )

    @pytest.fixture
    def index_client(self, exporter):
        """Create a SearchIndexClient for testing."""
        return SearchIndexClient(
            endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]),
        )

    @pytest.fixture(autouse=True)
    def clear_exporter_before_test(self, exporter):
        """Clear exporter before each test."""
        exporter.clear()
        yield

    @pytest.mark.vcr
    def test_search(self, exporter, search_client):
        """Test that search() creates a span with correct attributes."""
        list(search_client.search(
            search_text="hotel",
            top=5,
            filter="rating ge 3",
        ))

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.search"
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Azure AI Search"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == INTEGRATION_TEST_INDEX
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_TEXT) == "hotel"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_TOP) == 5
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_FILTER) == "rating ge 3"

    @pytest.mark.vcr
    def test_search_with_skip(self, exporter, search_client):
        """Test that search() captures skip parameter."""
        list(search_client.search(
            search_text="*",
            top=10,
            skip=5,
        ))

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.search"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_SKIP) == 5

    @pytest.mark.vcr
    def test_get_document(self, exporter, search_client):
        """Test that get_document() creates a span with document key."""
        # First upload a document
        search_client.upload_documents([{"id": "doc-1", "name": "Test", "description": "Test", "rating": 4.0}])
        exporter.clear()

        # Give Azure time to index the document
        import time
        time.sleep(1)

        try:
            search_client.get_document(key="doc-1")
        except Exception:
            pass  # Document may not be indexed yet, but span should still be created

        spans = exporter.get_finished_spans()
        assert len(spans) >= 1

        # Find the get_document span
        get_doc_spans = [s for s in spans if s.name == "azure_search.get_document"]
        assert len(get_doc_spans) == 1

        span = get_doc_spans[0]
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_KEY) == "doc-1"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == INTEGRATION_TEST_INDEX

        # Verify content attribute exists for the document response
        assert EventAttributes.DB_QUERY_RESULT_DOCUMENT.value in dict(span.attributes)

    @pytest.mark.vcr
    def test_get_document_count(self, exporter, search_client):
        """Test that get_document_count() creates a span."""
        search_client.get_document_count()

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.get_document_count"
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == INTEGRATION_TEST_INDEX

    @pytest.mark.vcr
    def test_upload_documents(self, exporter, search_client):
        """Test that upload_documents() creates a span with document count."""
        documents = [
            {"id": "test-1", "name": "Test Hotel 1", "description": "A test hotel", "rating": 4.0},
            {"id": "test-2", "name": "Test Hotel 2", "description": "Another test hotel", "rating": 3.5},
        ]
        search_client.upload_documents(documents=documents)

        spans = exporter.get_finished_spans()
        upload_spans = [s for s in spans if s.name == "azure_search.upload_documents"]
        assert len(upload_spans) == 1

        span = upload_spans[0]
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 2

        # Verify request-side content attributes (one per document)
        attrs = dict(span.attributes)
        assert f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.0" in attrs
        assert f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.1" in attrs

        # Verify response-side content attributes (one per IndexingResult)
        assert f"{EventAttributes.DB_QUERY_RESULT_METADATA.value}.0" in attrs

    @pytest.mark.vcr
    def test_merge_documents(self, exporter, search_client):
        """Test that merge_documents() creates a span with document count."""
        # First upload a document to merge
        search_client.upload_documents([{"id": "merge-1", "name": "Merge Test", "description": "Test", "rating": 3.0}])
        exporter.clear()

        documents = [
            {"id": "merge-1", "rating": 4.8},
        ]
        search_client.merge_documents(documents=documents)

        spans = exporter.get_finished_spans()
        merge_spans = [s for s in spans if s.name == "azure_search.merge_documents"]
        assert len(merge_spans) == 1

        span = merge_spans[0]
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 1

    @pytest.mark.vcr
    def test_delete_documents(self, exporter, search_client):
        """Test that delete_documents() creates a span with document count."""
        documents = [
            {"id": "test-1"},
            {"id": "test-2"},
        ]
        search_client.delete_documents(documents=documents)

        spans = exporter.get_finished_spans()
        delete_spans = [s for s in spans if s.name == "azure_search.delete_documents"]
        assert len(delete_spans) == 1

        span = delete_spans[0]
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 2

    @pytest.mark.vcr
    def test_merge_or_upload_documents(self, exporter, search_client):
        """Test that merge_or_upload_documents() creates a span with document count."""
        documents = [
            {"id": "upsert-1", "name": "Upsert Hotel", "description": "A test upsert", "rating": 4.2},
        ]
        search_client.merge_or_upload_documents(documents=documents)

        spans = exporter.get_finished_spans()
        upsert_spans = [s for s in spans if s.name == "azure_search.merge_or_upload_documents"]
        assert len(upsert_spans) == 1

        span = upsert_spans[0]
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 1

    @pytest.mark.vcr
    def test_index_documents(self, exporter, search_client):
        """Test that index_documents() creates a span with batch size."""
        batch = IndexDocumentsBatch()
        batch.add_upload_actions([
            {"id": "batch-1", "name": "Batch Hotel", "description": "A batch test", "rating": 3.9},
        ])
        search_client.index_documents(batch=batch)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.index_documents"
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 1

    @pytest.mark.vcr
    def test_autocomplete(self, exporter, search_client):
        """Test that autocomplete() creates a span with search text and suggester."""
        # Upload some documents first for autocomplete to work
        search_client.upload_documents([
            {"id": "auto-1", "name": "Luxury Hotel", "description": "A luxury hotel", "rating": 5.0},
        ])
        exporter.clear()

        import time
        time.sleep(2)  # Wait for indexing

        try:
            list(search_client.autocomplete(
                search_text="lux",
                suggester_name="sg",
            ))
        except Exception:
            pass  # May fail if index not ready, but span should still be created

        spans = exporter.get_finished_spans()
        autocomplete_spans = [s for s in spans if s.name == "azure_search.autocomplete"]
        assert len(autocomplete_spans) == 1

        span = autocomplete_spans[0]
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_TEXT) == "lux"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SUGGESTER_NAME) == "sg"

        # Verify content attributes exist for autocomplete results (if any results returned)
        # Attributes should be present if results were returned; may be absent if index wasn't ready
        attrs = dict(span.attributes)
        entity_keys = [k for k in attrs if k.startswith(EventAttributes.DB_SEARCH_RESULT_ENTITY.value + ".")]
        # Each entity key should have a value
        for key in entity_keys:
            assert attrs[key] is not None

    @pytest.mark.vcr
    def test_suggest(self, exporter, search_client):
        """Test that suggest() creates a span with search text and suggester."""
        # Upload some documents first for suggest to work
        search_client.upload_documents([
            {"id": "sug-1", "name": "Hot Springs Resort", "description": "A hot springs resort", "rating": 4.5},
        ])
        exporter.clear()

        import time
        time.sleep(2)  # Wait for indexing

        try:
            list(search_client.suggest(
                search_text="hot",
                suggester_name="sg",
            ))
        except Exception:
            pass  # May fail if index not ready, but span should still be created

        spans = exporter.get_finished_spans()
        suggest_spans = [s for s in spans if s.name == "azure_search.suggest"]
        assert len(suggest_spans) == 1

        span = suggest_spans[0]
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_TEXT) == "hot"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SUGGESTER_NAME) == "sg"

        # Verify content attributes exist for suggest results (if any results returned)
        attrs = dict(span.attributes)
        entity_keys = [k for k in attrs if k.startswith(EventAttributes.DB_SEARCH_RESULT_ENTITY.value + ".")]
        for key in entity_keys:
            assert attrs[key] is not None

    @pytest.mark.vcr
    def test_content_disabled_no_content_attributes(self, exporter, search_client, monkeypatch):
        """Test that no content attributes are set when TRACELOOP_TRACE_CONTENT=false."""
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

        search_client.get_document_count()

        content_attr_prefixes = (
            EventAttributes.DB_QUERY_RESULT_DOCUMENT.value,
            EventAttributes.DB_SEARCH_RESULT_ENTITY.value,
            EventAttributes.DB_SEARCH_EMBEDDINGS_VECTOR.value,
            EventAttributes.DB_QUERY_RESULT_METADATA.value,
            EventAttributes.DB_QUERY_RESULT_ID.value,
        )

        spans = exporter.get_finished_spans()
        for span in spans:
            for attr_key in dict(span.attributes):
                for prefix in content_attr_prefixes:
                    assert not attr_key.startswith(prefix) or attr_key == prefix, (
                        f"Found content attribute {attr_key} with content tracing disabled"
                    )


class TestSearchIndexClientIntegration:
    """Integration tests for SearchIndexClient instrumentation."""

    @pytest.fixture(scope="class")
    def index_client_setup(self):
        """Create a SearchIndexClient for setting up test index."""
        return SearchIndexClient(
            endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]),
        )

    @pytest.fixture(scope="class", autouse=True)
    def setup_test_index(self, index_client_setup):
        """Set up the test index with required schema before all tests, tear down after."""
        # Skip setup/teardown in playback mode (when using test credentials)
        is_playback_mode = os.environ.get("AZURE_SEARCH_ADMIN_KEY") == "test-api-key"

        if not is_playback_mode:
            # Define index schema
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="name", type=SearchFieldDataType.String),
                SearchableField(name="description", type=SearchFieldDataType.String),
                SimpleField(name="rating", type=SearchFieldDataType.Double, filterable=True),
            ]
            suggesters = [SearchSuggester(name="sg", source_fields=["name"])]
            index = SearchIndex(name=INTEGRATION_TEST_INDEX, fields=fields, suggesters=suggesters)

            # Delete index if it exists, then create fresh
            try:
                index_client_setup.delete_index(INTEGRATION_TEST_INDEX)
            except Exception:
                pass  # Index may not exist

            # Create the index
            index_client_setup.create_index(index)

        yield

        if not is_playback_mode:
            # Teardown: Delete the index and all documents after all tests
            try:
                index_client_setup.delete_index(INTEGRATION_TEST_INDEX)
            except Exception:
                pass  # Clean up best effort

    @pytest.fixture
    def index_client(self, exporter):
        """Create a SearchIndexClient for testing."""
        return SearchIndexClient(
            endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]),
        )

    @pytest.fixture(autouse=True)
    def clear_exporter_before_test(self, exporter):
        """Clear exporter before each test."""
        exporter.clear()
        yield

    @pytest.mark.vcr
    def test_list_indexes(self, exporter, index_client):
        """Test that list_indexes() creates a span."""
        list(index_client.list_indexes())

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.list_indexes"
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Azure AI Search"

    @pytest.mark.vcr
    def test_get_index(self, exporter, index_client):
        """Test that get_index() creates a span with index name."""
        index_client.get_index(INTEGRATION_TEST_INDEX)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.get_index"
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == INTEGRATION_TEST_INDEX

    @pytest.mark.vcr
    def test_get_index_statistics(self, exporter, index_client):
        """Test that get_index_statistics() creates a span with index name."""
        index_client.get_index_statistics(INTEGRATION_TEST_INDEX)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.get_index_statistics"
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == INTEGRATION_TEST_INDEX

    @pytest.mark.vcr
    def test_create_index(self, exporter, index_client):
        """Test that create_index() creates a span with index name."""
        test_index_name = "test-create-index"

        # Clean up if exists
        try:
            index_client.delete_index(test_index_name)
            exporter.clear()
        except Exception:
            exporter.clear()

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
        ]
        index = SearchIndex(name=test_index_name, fields=fields)
        index_client.create_index(index)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.create_index"
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == test_index_name

        # Cleanup
        try:
            index_client.delete_index(test_index_name)
        except Exception:
            pass

    @pytest.mark.vcr
    def test_create_or_update_index(self, exporter, index_client):
        """Test that create_or_update_index() creates a span with index name."""
        test_index_name = "test-upsert-index"

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
        ]
        index = SearchIndex(name=test_index_name, fields=fields)
        index_client.create_or_update_index(index)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.create_or_update_index"
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == test_index_name

        # Cleanup
        try:
            index_client.delete_index(test_index_name)
        except Exception:
            pass

    @pytest.mark.vcr
    def test_delete_index(self, exporter, index_client):
        """Test that delete_index() creates a span with index name."""
        test_index_name = "test-delete-index"

        # Create index first
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        ]
        index = SearchIndex(name=test_index_name, fields=fields)
        try:
            index_client.create_index(index)
            exporter.clear()
        except Exception:
            exporter.clear()

        index_client.delete_index(test_index_name)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.delete_index"
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == test_index_name

    @pytest.mark.vcr
    def test_analyze_text(self, exporter, index_client):
        """Test that analyze_text() creates a span with index and analyzer name."""
        analyze_request = AnalyzeTextOptions(
            text="The quick brown fox",
            analyzer_name="standard.lucene",
        )
        index_client.analyze_text(
            index_name=INTEGRATION_TEST_INDEX,
            analyze_request=analyze_request,
        )

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.analyze_text"
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == INTEGRATION_TEST_INDEX
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_ANALYZER_NAME) == "standard.lucene"

    @pytest.mark.vcr
    def test_get_service_statistics(self, exporter, index_client):
        """Test that get_service_statistics() creates a span with service counters."""
        index_client.get_service_statistics()

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.get_service_statistics"
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Azure AI Search"
        # Service statistics response attributes are set from response
        # document_count may be 0, so check key exists rather than is not None
        assert SpanAttributes.AZURE_SEARCH_SERVICE_DOCUMENT_COUNT in span.attributes
        assert SpanAttributes.AZURE_SEARCH_SERVICE_INDEX_COUNT in span.attributes

    @pytest.mark.vcr
    def test_list_index_names(self, exporter, index_client):
        """Test that list_index_names() creates a span."""
        list(index_client.list_index_names())

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.list_index_names"
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Azure AI Search"


class TestSynonymMapIntegration:
    """Integration tests for synonym map operations."""

    @pytest.fixture(scope="class")
    def index_client_setup(self):
        """Create a SearchIndexClient for synonym map operations."""
        return SearchIndexClient(
            endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]),
        )

    @pytest.fixture
    def index_client(self, exporter):
        """Create a SearchIndexClient for testing."""
        return SearchIndexClient(
            endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]),
        )

    @pytest.fixture(autouse=True)
    def clear_exporter_before_test(self, exporter):
        """Clear exporter before each test."""
        exporter.clear()
        yield

    @pytest.mark.vcr
    def test_create_synonym_map(self, exporter, index_client):
        """Test that create_synonym_map() creates a span with synonym map attributes."""
        sm_name = "otel-test-synonyms"

        # Clean up if exists
        try:
            index_client.delete_synonym_map(sm_name)
            exporter.clear()
        except Exception:
            exporter.clear()

        synonym_map = SynonymMap(name=sm_name, synonyms=["hotel,motel", "cozy,comfortable,warm"])
        index_client.create_synonym_map(synonym_map)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.create_synonym_map"
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Azure AI Search"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_NAME) == sm_name
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_SYNONYMS_COUNT) == 2

        # Cleanup
        try:
            index_client.delete_synonym_map(sm_name)
        except Exception:
            pass

    @pytest.mark.vcr
    def test_get_synonym_map(self, exporter, index_client):
        """Test that get_synonym_map() creates a span with synonym map name."""
        sm_name = "otel-test-get-sm"

        # Create synonym map first
        synonym_map = SynonymMap(name=sm_name, synonyms=["big,large"])
        try:
            index_client.create_synonym_map(synonym_map)
            exporter.clear()
        except Exception:
            exporter.clear()

        index_client.get_synonym_map(sm_name)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.get_synonym_map"
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_NAME) == sm_name

        # Cleanup
        try:
            index_client.delete_synonym_map(sm_name)
        except Exception:
            pass

    @pytest.mark.vcr
    def test_get_synonym_maps(self, exporter, index_client):
        """Test that get_synonym_maps() creates a span."""
        sm_name = "otel-test-list-sms"

        # Create synonym map first so the listing returns non-empty results
        synonym_map = SynonymMap(name=sm_name, synonyms=["hello,hi", "goodbye,bye"])
        try:
            index_client.create_synonym_map(synonym_map)
            exporter.clear()
        except Exception:
            exporter.clear()

        index_client.get_synonym_maps()

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.get_synonym_maps"
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Azure AI Search"

        # Cleanup
        try:
            index_client.delete_synonym_map(sm_name)
        except Exception:
            pass

    @pytest.mark.vcr
    def test_create_or_update_synonym_map(self, exporter, index_client):
        """Test that create_or_update_synonym_map() creates a span."""
        sm_name = "otel-test-upsert-sm"

        synonym_map = SynonymMap(name=sm_name, synonyms=["fast,quick", "slow,sluggish"])
        index_client.create_or_update_synonym_map(synonym_map)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.create_or_update_synonym_map"
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_NAME) == sm_name
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_SYNONYMS_COUNT) == 2

        # Cleanup
        try:
            index_client.delete_synonym_map(sm_name)
        except Exception:
            pass

    @pytest.mark.vcr
    def test_delete_synonym_map(self, exporter, index_client):
        """Test that delete_synonym_map() creates a span with synonym map name."""
        sm_name = "otel-test-delete-sm"

        # Create synonym map first
        synonym_map = SynonymMap(name=sm_name, synonyms=["old,ancient"])
        try:
            index_client.create_synonym_map(synonym_map)
            exporter.clear()
        except Exception:
            exporter.clear()

        index_client.delete_synonym_map(sm_name)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.delete_synonym_map"
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_NAME) == sm_name

    @pytest.mark.vcr
    def test_get_synonym_map_names(self, exporter, index_client):
        """Test that get_synonym_map_names() creates a span."""
        sm_name = "otel-test-list-sm-names"

        # Create synonym map first so the listing returns non-empty results
        synonym_map = SynonymMap(name=sm_name, synonyms=["warm,cozy"])
        try:
            index_client.create_synonym_map(synonym_map)
            exporter.clear()
        except Exception:
            exporter.clear()

        index_client.get_synonym_map_names()

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.get_synonym_map_names"
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Azure AI Search"

        # Cleanup
        try:
            index_client.delete_synonym_map(sm_name)
        except Exception:
            pass


class TestSearchIndexerClientIntegration:
    """Integration tests for SearchIndexerClient name-only listing methods."""

    @pytest.fixture
    def indexer_client(self, exporter):
        """Create a SearchIndexerClient for testing."""
        return SearchIndexerClient(
            endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]),
        )

    @pytest.fixture(autouse=True)
    def clear_exporter_before_test(self, exporter):
        """Clear exporter before each test."""
        exporter.clear()
        yield

    @pytest.mark.vcr
    def test_get_indexer_names(self, exporter, indexer_client):
        """Test that get_indexer_names() creates a span."""
        ds_name = "otel-test-indexer-ds"
        indexer_name = "otel-test-indexer-names"

        # Create a data source connection and indexer so the listing returns non-empty results
        ds_connection = SearchIndexerDataSourceConnection(
            name=ds_name,
            type="azureblob",
            connection_string=os.environ.get(
                "AZURE_STORAGE_CONNECTION_STRING",
                "DefaultEndpointsProtocol=https;AccountName=placeholder;AccountKey=placeholder;EndpointSuffix=core.windows.net",
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
            exporter.clear()
        except Exception:
            exporter.clear()

        list(indexer_client.get_indexer_names())

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.get_indexer_names"
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Azure AI Search"

        # Cleanup
        try:
            indexer_client.delete_indexer(indexer_name)
        except Exception:
            pass
        try:
            indexer_client.delete_data_source_connection(ds_name)
        except Exception:
            pass

    @pytest.mark.vcr
    def test_get_data_source_connection_names(self, exporter, indexer_client):
        """Test that get_data_source_connection_names() creates a span."""
        ds_name = "otel-test-ds-names"

        # Create a data source connection first so the listing returns non-empty results
        ds_connection = SearchIndexerDataSourceConnection(
            name=ds_name,
            type="azureblob",
            connection_string=os.environ.get(
                "AZURE_STORAGE_CONNECTION_STRING",
                "DefaultEndpointsProtocol=https;AccountName=placeholder;AccountKey=placeholder;EndpointSuffix=core.windows.net",
            ),
            container=SearchIndexerDataContainer(name="placeholder-container"),
        )
        try:
            indexer_client.create_data_source_connection(ds_connection)
            exporter.clear()
        except Exception:
            exporter.clear()

        list(indexer_client.get_data_source_connection_names())

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.get_data_source_connection_names"
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Azure AI Search"

        # Cleanup
        try:
            indexer_client.delete_data_source_connection(ds_name)
        except Exception:
            pass

    @pytest.mark.vcr
    def test_get_skillset_names(self, exporter, indexer_client):
        """Test that get_skillset_names() creates a span."""
        skillset_name = "otel-test-skillset-names"

        # Create a skillset first so the listing returns non-empty results
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
            exporter.clear()
        except Exception:
            exporter.clear()

        list(indexer_client.get_skillset_names())

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.get_skillset_names"
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Azure AI Search"

        # Cleanup
        try:
            indexer_client.delete_skillset(skillset_name)
        except Exception:
            pass
