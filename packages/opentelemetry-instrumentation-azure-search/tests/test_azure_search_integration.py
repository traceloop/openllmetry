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
)
from azure.search.documents import IndexDocumentsBatch
from opentelemetry.semconv_ai import SpanAttributes
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
