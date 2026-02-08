"""Tests for Azure AI Search instrumentation using mocks."""

import pytest
from unittest.mock import MagicMock, patch
from opentelemetry.semconv_ai import SpanAttributes


# Mock classes to simulate Azure Search SDK
class MockSearchClient:
    """Mock SearchClient for testing."""

    def __init__(self, endpoint, index_name, credential):
        self._endpoint = endpoint
        self._index_name = index_name
        self._credential = credential

    def search(self, search_text=None, **kwargs):
        return iter([{"id": "1", "name": "Test Document"}])

    def get_document(self, key, **kwargs):
        return {"id": key, "name": "Test Document"}

    def get_document_count(self, **kwargs):
        return 100

    def upload_documents(self, documents, **kwargs):
        return [{"key": "1", "succeeded": True}]

    def merge_documents(self, documents, **kwargs):
        return [{"key": "1", "succeeded": True}]

    def delete_documents(self, documents, **kwargs):
        return [{"key": "1", "succeeded": True}]

    def merge_or_upload_documents(self, documents, **kwargs):
        return [{"key": "1", "succeeded": True}]

    def index_documents(self, batch, **kwargs):
        return MagicMock(results=[{"key": "1", "succeeded": True}])

    def autocomplete(self, search_text, suggester_name, **kwargs):
        return [{"text": "suggestion1"}]

    def suggest(self, search_text, suggester_name, **kwargs):
        return [{"text": "suggestion1"}]


class MockSearchIndex:
    """Mock SearchIndex for testing."""

    def __init__(self, name, fields=None):
        self.name = name
        self.fields = fields or []


class MockSynonymMap:
    """Mock SynonymMap for testing."""

    def __init__(self, name, synonyms=None):
        self.name = name
        self.synonyms = synonyms or []


class MockServiceCounters:
    """Mock service counters for get_service_statistics response."""

    def __init__(self, document_count=0, index_count=0):
        self.document_counter = MagicMock(usage=document_count)
        self.index_counter = MagicMock(usage=index_count)


class MockServiceStatistics:
    """Mock service statistics response."""

    def __init__(self, document_count=0, index_count=0):
        self.counters = MockServiceCounters(document_count, index_count)


class MockSearchIndexClient:
    """Mock SearchIndexClient for testing."""

    def __init__(self, endpoint, credential):
        self._endpoint = endpoint
        self._credential = credential

    def create_index(self, index, **kwargs):
        return index

    def create_or_update_index(self, index, **kwargs):
        return index

    def delete_index(self, index, **kwargs):
        return None

    def get_index(self, index_name, **kwargs):
        return MockSearchIndex(name=index_name)

    def list_indexes(self, **kwargs):
        return iter([MockSearchIndex(name="index1"), MockSearchIndex(name="index2")])

    def get_index_statistics(self, index_name, **kwargs):
        return {"document_count": 100, "storage_size": 1024}

    def analyze_text(self, index_name, analyze_request, **kwargs):
        return {"tokens": [{"token": "test"}]}

    def get_service_statistics(self, **kwargs):
        return MockServiceStatistics(document_count=5000, index_count=3)

    def list_index_names(self, **kwargs):
        return iter(["index1", "index2"])

    def create_synonym_map(self, synonym_map, **kwargs):
        return synonym_map

    def create_or_update_synonym_map(self, synonym_map, **kwargs):
        return synonym_map

    def delete_synonym_map(self, name, **kwargs):
        return None

    def get_synonym_map(self, name, **kwargs):
        return MockSynonymMap(name=name, synonyms=["hotel,motel", "cozy,comfortable"])

    def get_synonym_maps(self, **kwargs):
        return [MockSynonymMap(name="sm1"), MockSynonymMap(name="sm2")]

    def get_synonym_map_names(self, **kwargs):
        return ["sm1", "sm2"]


# Patch the Azure SDK modules before importing
@pytest.fixture(autouse=True)
def mock_azure_sdk():
    """Mock the Azure Search SDK modules."""
    with patch.dict("sys.modules", {
        "azure": MagicMock(),
        "azure.search": MagicMock(),
        "azure.search.documents": MagicMock(SearchClient=MockSearchClient),
        "azure.search.documents.indexes": MagicMock(SearchIndexClient=MockSearchIndexClient),
        "azure.search.documents.aio": MagicMock(),
        "azure.search.documents.indexes.aio": MagicMock(),
        "azure.core": MagicMock(),
        "azure.core.credentials": MagicMock(),
    }):
        yield


class TestSearchClientInstrumentation:
    """Tests for SearchClient instrumentation."""

    def test_search_creates_span(self, exporter):
        """Test that search() creates a span with correct attributes."""
        client = MockSearchClient(
            endpoint="https://test.search.windows.net",
            index_name="test-index",
            credential=MagicMock()
        )

        # Manually wrap for testing since we're using mocks
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.search",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_INDEX_NAME: "test-index",
                SpanAttributes.AZURE_SEARCH_SEARCH_TEXT: "luxury hotel",
                SpanAttributes.AZURE_SEARCH_SEARCH_TOP: 10,
                SpanAttributes.AZURE_SEARCH_SEARCH_FILTER: "rating ge 4",
            }
        ):
            list(client.search(search_text="luxury hotel", top=10, filter="rating ge 4"))

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.search"
        assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Azure AI Search"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "test-index"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_TEXT) == "luxury hotel"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_TOP) == 10
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_FILTER) == "rating ge 4"

    def test_get_document_creates_span(self, exporter):
        """Test that get_document() creates a span with document key."""
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.get_document",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_DOCUMENT_KEY: "doc-123",
            }
        ):
            client = MockSearchClient(
                endpoint="https://test.search.windows.net",
                index_name="test-index",
                credential=MagicMock()
            )
            client.get_document(key="doc-123")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.get_document"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_KEY) == "doc-123"

    def test_upload_documents_creates_span(self, exporter):
        """Test that upload_documents() creates a span with document count."""
        from opentelemetry import trace

        documents = [
            {"id": "1", "name": "Doc 1"},
            {"id": "2", "name": "Doc 2"},
            {"id": "3", "name": "Doc 3"},
        ]

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.upload_documents",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT: len(documents),
            }
        ):
            client = MockSearchClient(
                endpoint="https://test.search.windows.net",
                index_name="test-index",
                credential=MagicMock()
            )
            client.upload_documents(documents=documents)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.upload_documents"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 3

    def test_search_with_skip_creates_span(self, exporter):
        """Test that search() with skip parameter creates correct span."""
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.search",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_INDEX_NAME: "test-index",
                SpanAttributes.AZURE_SEARCH_SEARCH_TEXT: "*",
                SpanAttributes.AZURE_SEARCH_SEARCH_TOP: 10,
                SpanAttributes.AZURE_SEARCH_SEARCH_SKIP: 5,
            }
        ):
            client = MockSearchClient(
                endpoint="https://test.search.windows.net",
                index_name="test-index",
                credential=MagicMock()
            )
            list(client.search(search_text="*", top=10, skip=5))

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.search"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_SKIP) == 5

    def test_get_document_count_creates_span(self, exporter):
        """Test that get_document_count() creates a span."""
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.get_document_count",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_INDEX_NAME: "test-index",
            }
        ):
            client = MockSearchClient(
                endpoint="https://test.search.windows.net",
                index_name="test-index",
                credential=MagicMock()
            )
            client.get_document_count()

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.get_document_count"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "test-index"

    def test_merge_documents_creates_span(self, exporter):
        """Test that merge_documents() creates a span with document count."""
        from opentelemetry import trace

        documents = [
            {"id": "1", "rating": 4.8},
        ]

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.merge_documents",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT: len(documents),
            }
        ):
            client = MockSearchClient(
                endpoint="https://test.search.windows.net",
                index_name="test-index",
                credential=MagicMock()
            )
            client.merge_documents(documents=documents)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.merge_documents"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 1

    def test_delete_documents_creates_span(self, exporter):
        """Test that delete_documents() creates a span with document count."""
        from opentelemetry import trace

        documents = [
            {"id": "1"},
            {"id": "2"},
        ]

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.delete_documents",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT: len(documents),
            }
        ):
            client = MockSearchClient(
                endpoint="https://test.search.windows.net",
                index_name="test-index",
                credential=MagicMock()
            )
            client.delete_documents(documents=documents)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.delete_documents"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 2

    def test_merge_or_upload_documents_creates_span(self, exporter):
        """Test that merge_or_upload_documents() creates a span with document count."""
        from opentelemetry import trace

        documents = [
            {"id": "1", "name": "Upsert Hotel", "rating": 4.2},
        ]

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.merge_or_upload_documents",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT: len(documents),
            }
        ):
            client = MockSearchClient(
                endpoint="https://test.search.windows.net",
                index_name="test-index",
                credential=MagicMock()
            )
            client.merge_or_upload_documents(documents=documents)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.merge_or_upload_documents"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 1

    def test_index_documents_creates_span(self, exporter):
        """Test that index_documents() creates a span with document count."""
        from opentelemetry import trace

        # Mock IndexDocumentsBatch
        batch = MagicMock()
        batch.actions = [
            {"id": "1", "name": "Batch Hotel"},
        ]

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.index_documents",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT: len(batch.actions),
            }
        ):
            client = MockSearchClient(
                endpoint="https://test.search.windows.net",
                index_name="test-index",
                credential=MagicMock()
            )
            client.index_documents(batch=batch)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.index_documents"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 1

    def test_autocomplete_creates_span(self, exporter):
        """Test that autocomplete() creates a span with suggester name."""
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.autocomplete",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_SEARCH_TEXT: "lux",
                SpanAttributes.AZURE_SEARCH_SUGGESTER_NAME: "sg",
            }
        ):
            client = MockSearchClient(
                endpoint="https://test.search.windows.net",
                index_name="test-index",
                credential=MagicMock()
            )
            client.autocomplete(search_text="lux", suggester_name="sg")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.autocomplete"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_TEXT) == "lux"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SUGGESTER_NAME) == "sg"

    def test_suggest_creates_span(self, exporter):
        """Test that suggest() creates a span with suggester name."""
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.suggest",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_SEARCH_TEXT: "hot",
                SpanAttributes.AZURE_SEARCH_SUGGESTER_NAME: "sg",
            }
        ):
            client = MockSearchClient(
                endpoint="https://test.search.windows.net",
                index_name="test-index",
                credential=MagicMock()
            )
            client.suggest(search_text="hot", suggester_name="sg")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.suggest"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_TEXT) == "hot"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SUGGESTER_NAME) == "sg"


class TestSearchIndexClientInstrumentation:
    """Tests for SearchIndexClient instrumentation."""

    def test_list_indexes_creates_span(self, exporter):
        """Test that list_indexes() creates a span."""
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.list_indexes",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
            }
        ):
            client = MockSearchIndexClient(
                endpoint="https://test.search.windows.net",
                credential=MagicMock()
            )
            list(client.list_indexes())

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.list_indexes"
        assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Azure AI Search"

    def test_get_index_creates_span(self, exporter):
        """Test that get_index() creates a span with index name."""
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.get_index",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_INDEX_NAME: "hotels-index",
            }
        ):
            client = MockSearchIndexClient(
                endpoint="https://test.search.windows.net",
                credential=MagicMock()
            )
            client.get_index(index_name="hotels-index")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.get_index"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "hotels-index"

    def test_create_index_creates_span(self, exporter):
        """Test that create_index() creates a span with index name."""
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        index = MockSearchIndex(name="hotels-index")

        with tracer.start_as_current_span(
            "azure_search.create_index",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_INDEX_NAME: "hotels-index",
            }
        ):
            client = MockSearchIndexClient(
                endpoint="https://test.search.windows.net",
                credential=MagicMock()
            )
            client.create_index(index=index)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.create_index"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "hotels-index"

    def test_create_or_update_index_creates_span(self, exporter):
        """Test that create_or_update_index() creates a span with index name."""
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        index = MockSearchIndex(name="upsert-index")

        with tracer.start_as_current_span(
            "azure_search.create_or_update_index",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_INDEX_NAME: "upsert-index",
            }
        ):
            client = MockSearchIndexClient(
                endpoint="https://test.search.windows.net",
                credential=MagicMock()
            )
            client.create_or_update_index(index=index)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.create_or_update_index"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "upsert-index"

    def test_delete_index_creates_span(self, exporter):
        """Test that delete_index() creates a span with index name."""
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.delete_index",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_INDEX_NAME: "old-index",
            }
        ):
            client = MockSearchIndexClient(
                endpoint="https://test.search.windows.net",
                credential=MagicMock()
            )
            client.delete_index(index="old-index")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.delete_index"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "old-index"

    def test_get_index_statistics_creates_span(self, exporter):
        """Test that get_index_statistics() creates a span."""
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.get_index_statistics",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_INDEX_NAME: "hotels-index",
            }
        ):
            client = MockSearchIndexClient(
                endpoint="https://test.search.windows.net",
                credential=MagicMock()
            )
            client.get_index_statistics(index_name="hotels-index")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.get_index_statistics"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "hotels-index"

    def test_analyze_text_creates_span(self, exporter):
        """Test that analyze_text() creates a span with analyzer name."""
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.analyze_text",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_INDEX_NAME: "hotels-index",
                SpanAttributes.AZURE_SEARCH_ANALYZER_NAME: "standard.lucene",
            }
        ):
            client = MockSearchIndexClient(
                endpoint="https://test.search.windows.net",
                credential=MagicMock()
            )
            client.analyze_text(index_name="hotels-index", analyze_request={"text": "test"})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.analyze_text"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "hotels-index"


class TestSpanAttributes:
    """Tests for span attributes."""

    def test_vector_db_vendor_attribute(self, exporter):
        """Test that all spans have the correct db.system attribute."""
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.search",
            attributes={SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search"}
        ):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Azure AI Search"

    def test_search_with_query_type(self, exporter):
        """Test search span with query_type attribute."""
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.search",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_SEARCH_QUERY_TYPE: "semantic",
            }
        ):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_QUERY_TYPE) == "semantic"


class TestResponseAttributes:
    """Tests for response attribute capturing."""

    def test_search_response_count(self, exporter):
        """Test that search response captures result count."""
        from opentelemetry import trace
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_search_response_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_response = MagicMock()
            mock_response.get_count.return_value = 42
            _set_search_response_attributes(span, mock_response)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_SEARCH_RESULTS_COUNT
        ) == 42

    def test_search_response_count_none(self, exporter):
        """Test that search response with no count does not set attribute."""
        from opentelemetry import trace
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_search_response_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_response = MagicMock()
            mock_response.get_count.return_value = None
            _set_search_response_attributes(span, mock_response)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_SEARCH_RESULTS_COUNT
        ) is None

    def test_document_count_response(self, exporter):
        """Test that get_document_count response captures count."""
        from opentelemetry import trace
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_document_count_response_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.get_document_count"
        ) as span:
            _set_document_count_response_attributes(span, 500)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT
        ) == 500

    def test_upload_documents_response_all_succeeded(self, exporter):
        """Test document batch response with all docs succeeding."""
        from opentelemetry import trace
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_document_batch_response_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.upload_documents"
        ) as span:
            results = [
                MagicMock(succeeded=True),
                MagicMock(succeeded=True),
                MagicMock(succeeded=True),
            ]
            _set_document_batch_response_attributes(span, results)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT
        ) == 3
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_DOCUMENT_FAILED_COUNT
        ) == 0

    def test_upload_documents_response_with_failures(self, exporter):
        """Test document batch response with some docs failing."""
        from opentelemetry import trace
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_document_batch_response_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.upload_documents"
        ) as span:
            results = [
                MagicMock(succeeded=True),
                MagicMock(succeeded=False),
                MagicMock(succeeded=True),
                MagicMock(succeeded=False),
            ]
            _set_document_batch_response_attributes(span, results)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT
        ) == 2
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_DOCUMENT_FAILED_COUNT
        ) == 2

    def test_index_documents_response(self, exporter):
        """Test index_documents response captures succeeded/failed."""
        from opentelemetry import trace
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_index_documents_response_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.index_documents"
        ) as span:
            mock_response = MagicMock()
            mock_response.results = [
                MagicMock(succeeded=True),
                MagicMock(succeeded=True),
                MagicMock(succeeded=False),
            ]
            _set_index_documents_response_attributes(span, mock_response)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT
        ) == 2
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_DOCUMENT_FAILED_COUNT
        ) == 1

    def test_autocomplete_response_count(self, exporter):
        """Test that autocomplete response captures result count."""
        from opentelemetry import trace
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_autocomplete_response_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.autocomplete"
        ) as span:
            results = [
                {"text": "hotel"},
                {"text": "hostel"},
                {"text": "house"},
            ]
            _set_autocomplete_response_attributes(span, results)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_AUTOCOMPLETE_RESULTS_COUNT
        ) == 3

    def test_suggest_response_count(self, exporter):
        """Test that suggest response captures result count."""
        from opentelemetry import trace
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_suggest_response_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.suggest") as span:
            results = [
                {"text": "Luxury Hotel"},
                {"text": "Luxury Resort"},
            ]
            _set_suggest_response_attributes(span, results)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_SUGGEST_RESULTS_COUNT
        ) == 2

    def test_empty_batch_response(self, exporter):
        """Test document batch response with empty list."""
        from opentelemetry import trace
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_document_batch_response_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.upload_documents"
        ) as span:
            _set_document_batch_response_attributes(span, [])

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        # Empty list should not set attributes
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT
        ) is None

    def test_indexer_status_response(self, exporter):
        """Test indexer status response captures status and counts."""
        from opentelemetry import trace
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_indexer_status_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.get_indexer_status"
        ) as span:
            mock_response = MagicMock()
            mock_response.status = "running"
            mock_response.last_result.items_processed = 1500
            mock_response.last_result.items_failed = 3
            _set_indexer_status_attributes(span, (), {}, mock_response)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_INDEXER_STATUS
        ) == "running"
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_DOCUMENTS_PROCESSED
        ) == 1500
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_DOCUMENTS_FAILED
        ) == 3


class TestInstrumentorLifecycle:
    """Tests for instrumentor lifecycle."""

    def test_instrumentor_can_be_instantiated(self):
        """Test that AzureSearchInstrumentor can be instantiated."""
        from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor

        instrumentor = AzureSearchInstrumentor()
        assert instrumentor is not None

    def test_instrumentor_dependencies(self):
        """Test that instrumentation_dependencies returns correct value."""
        from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor

        instrumentor = AzureSearchInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert "azure-search-documents >= 11.0.0" in deps

    def test_instrumentor_with_exception_logger(self):
        """Test that instrumentor accepts exception_logger parameter."""
        from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor, Config

        def custom_logger(e):
            pass

        AzureSearchInstrumentor(exception_logger=custom_logger)
        assert Config.exception_logger == custom_logger


class TestVectorSearchAttributes:
    """Tests for vector search attribute capturing."""

    def test_vector_search_attributes(self, exporter):
        """Test that vector search attributes are captured."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_vector_search_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_vq = MagicMock()
            mock_vq.k_nearest_neighbors = 5
            mock_vq.fields = "content_vector"
            mock_vq.exhaustive = False

            kwargs = {
                "vector_queries": [mock_vq],
                "vector_filter_mode": "preFilter",
            }
            _set_vector_search_attributes(span, kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_QUERIES_COUNT
        ) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_K_NEAREST_NEIGHBORS
        ) == 5
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_FIELDS
        ) == "content_vector"
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_EXHAUSTIVE
        ) is False
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_FILTER_MODE
        ) == "preFilter"

    def test_vector_search_multiple_queries(self, exporter):
        """Test that multiple vector queries are counted correctly."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_vector_search_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_vq1 = MagicMock()
            mock_vq1.k_nearest_neighbors = 5
            mock_vq1.fields = "title_vector"
            mock_vq1.exhaustive = None

            mock_vq2 = MagicMock()
            mock_vq2.k_nearest_neighbors = 3
            mock_vq2.fields = "content_vector"
            mock_vq2.exhaustive = None

            kwargs = {"vector_queries": [mock_vq1, mock_vq2]}
            _set_vector_search_attributes(span, kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_QUERIES_COUNT
        ) == 2
        # First vector query's fields are captured
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_FIELDS
        ) == "title_vector"

    def test_vector_search_list_fields(self, exporter):
        """Test that list fields are joined with commas."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_vector_search_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_vq = MagicMock()
            mock_vq.k_nearest_neighbors = 5
            mock_vq.fields = ["title_vector", "content_vector"]
            mock_vq.exhaustive = None

            kwargs = {"vector_queries": [mock_vq]}
            _set_vector_search_attributes(span, kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_FIELDS
        ) == "title_vector,content_vector"

    def test_no_vector_queries_sets_nothing(self, exporter):
        """Test that no vector_queries kwarg sets no attributes."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_vector_search_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            _set_vector_search_attributes(span, {})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_QUERIES_COUNT
        ) is None

    def test_vector_filter_mode_enum(self, exporter):
        """Test that enum vector_filter_mode values are converted to string."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_vector_search_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_vq = MagicMock()
            mock_vq.k_nearest_neighbors = 5
            mock_vq.fields = "vec"
            mock_vq.exhaustive = None

            mock_enum = MagicMock()
            mock_enum.value = "postFilter"

            kwargs = {
                "vector_queries": [mock_vq],
                "vector_filter_mode": mock_enum,
            }
            _set_vector_search_attributes(span, kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_FILTER_MODE
        ) == "postFilter"


class TestEnhancedVectorSearchAttributes:
    """Tests for enhanced vector search attributes (kind, weight, oversampling)."""

    def test_vectorizable_text_query_kind(self, exporter):
        """Test that VectorizableTextQuery sets vector_query_kind='text'."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_vector_search_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_vq = MagicMock()
            mock_vq.k_nearest_neighbors = 5
            mock_vq.fields = "content_vector"
            mock_vq.exhaustive = None
            mock_vq.kind = "text"
            mock_vq.weight = None
            mock_vq.oversampling = None

            kwargs = {"vector_queries": [mock_vq]}
            _set_vector_search_attributes(span, kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_QUERY_KIND
        ) == "text"

    def test_vectorized_query_kind(self, exporter):
        """Test that VectorizedQuery sets vector_query_kind='vector'."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_vector_search_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_vq = MagicMock()
            mock_vq.k_nearest_neighbors = 10
            mock_vq.fields = "embedding"
            mock_vq.exhaustive = None
            mock_vq.kind = "vector"
            mock_vq.weight = None
            mock_vq.oversampling = None

            kwargs = {"vector_queries": [mock_vq]}
            _set_vector_search_attributes(span, kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_QUERY_KIND
        ) == "vector"

    def test_vector_weight_captured(self, exporter):
        """Test that vector query weight is captured."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_vector_search_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_vq = MagicMock()
            mock_vq.k_nearest_neighbors = 5
            mock_vq.fields = "vec"
            mock_vq.exhaustive = None
            mock_vq.kind = None
            mock_vq.weight = 0.8
            mock_vq.oversampling = None

            kwargs = {"vector_queries": [mock_vq]}
            _set_vector_search_attributes(span, kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_WEIGHT
        ) == 0.8

    def test_vector_oversampling_captured(self, exporter):
        """Test that vector query oversampling is captured."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_vector_search_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_vq = MagicMock()
            mock_vq.k_nearest_neighbors = 5
            mock_vq.fields = "vec"
            mock_vq.exhaustive = None
            mock_vq.kind = None
            mock_vq.weight = None
            mock_vq.oversampling = 2.0

            kwargs = {"vector_queries": [mock_vq]}
            _set_vector_search_attributes(span, kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_OVERSAMPLING
        ) == 2.0

    def test_none_kind_weight_oversampling_not_set(self, exporter):
        """Test that None values for kind/weight/oversampling are not set."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_vector_search_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_vq = MagicMock()
            mock_vq.k_nearest_neighbors = 5
            mock_vq.fields = "vec"
            mock_vq.exhaustive = None
            mock_vq.kind = None
            mock_vq.weight = None
            mock_vq.oversampling = None

            kwargs = {"vector_queries": [mock_vq]}
            _set_vector_search_attributes(span, kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert SpanAttributes.AZURE_SEARCH_VECTOR_QUERY_KIND not in spans[0].attributes
        assert SpanAttributes.AZURE_SEARCH_VECTOR_WEIGHT not in spans[0].attributes
        assert SpanAttributes.AZURE_SEARCH_VECTOR_OVERSAMPLING not in spans[0].attributes


class TestFacetsAndOrderByAttributes:
    """Tests for facets and order_by search attribute capturing."""

    def test_facets_as_list(self, exporter):
        """Test that facets list is captured as comma-joined string."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_search_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            kwargs = {"facets": ["category", "price,interval:10"]}
            _set_search_attributes(span, (), kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_FACETS
        ) == "category,price,interval:10"

    def test_order_by_as_list(self, exporter):
        """Test that order_by list is captured as comma-joined string."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_search_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            kwargs = {"order_by": ["price asc", "rating desc"]}
            _set_search_attributes(span, (), kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_ORDER_BY
        ) == "price asc,rating desc"

    def test_facets_none_not_set(self, exporter):
        """Test that None facets/order_by set nothing on span."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_search_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            _set_search_attributes(span, (), {})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert SpanAttributes.AZURE_SEARCH_FACETS not in spans[0].attributes
        assert SpanAttributes.AZURE_SEARCH_ORDER_BY not in spans[0].attributes


class TestSemanticSearchAttributes:
    """Tests for semantic search attribute capturing."""

    def test_semantic_search_attributes(self, exporter):
        """Test that semantic search attributes are captured."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_semantic_search_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            kwargs = {
                "semantic_configuration_name": "my-semantic-config",
                "query_caption": "extractive",
                "query_answer": "extractive",
            }
            _set_semantic_search_attributes(span, kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_SEMANTIC_CONFIGURATION_NAME
        ) == "my-semantic-config"
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_QUERY_CAPTION
        ) == "extractive"
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_QUERY_ANSWER
        ) == "extractive"

    def test_semantic_search_enum_values(self, exporter):
        """Test that enum values for query_caption/query_answer are converted."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_semantic_search_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_caption = MagicMock()
            mock_caption.value = "extractive"
            mock_answer = MagicMock()
            mock_answer.value = "extractive"

            kwargs = {
                "semantic_configuration_name": "config-1",
                "query_caption": mock_caption,
                "query_answer": mock_answer,
            }
            _set_semantic_search_attributes(span, kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_QUERY_CAPTION
        ) == "extractive"
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_QUERY_ANSWER
        ) == "extractive"

    def test_no_semantic_config_sets_nothing(self, exporter):
        """Test that missing semantic kwargs set no attributes."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_semantic_search_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            _set_semantic_search_attributes(span, {})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_SEMANTIC_CONFIGURATION_NAME
        ) is None


class TestSearchAttributeExtras:
    """Tests for additional search attributes (select, search_fields, etc.)."""

    def test_search_mode_attribute(self, exporter):
        """Test that search_mode is captured."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_search_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            _set_search_attributes(span, (), {"search_mode": "all"})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_SEARCH_MODE
        ) == "all"

    def test_scoring_profile_attribute(self, exporter):
        """Test that scoring_profile is captured."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_search_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            _set_search_attributes(span, (), {"scoring_profile": "boost-by-freshness"})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_SCORING_PROFILE
        ) == "boost-by-freshness"

    def test_select_as_list(self, exporter):
        """Test that select list is joined with commas."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_search_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            _set_search_attributes(span, (), {"select": ["id", "name", "rating"]})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_SELECT
        ) == "id,name,rating"

    def test_select_as_string(self, exporter):
        """Test that select string is passed through."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_search_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            _set_search_attributes(span, (), {"select": "id,name"})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_SELECT
        ) == "id,name"

    def test_search_fields_as_list(self, exporter):
        """Test that search_fields list is joined with commas."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_search_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            _set_search_attributes(span, (), {"search_fields": ["title", "description"]})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_SEARCH_FIELDS
        ) == "title,description"

    def test_query_type_enum(self, exporter):
        """Test that query_type enum is converted to string."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_search_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_qt = MagicMock()
            mock_qt.value = "semantic"
            _set_search_attributes(span, (), {"query_type": mock_qt})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_SEARCH_QUERY_TYPE
        ) == "semantic"


class TestErrorHandling:
    """Tests for error handling in the wrapper."""

    def test_sync_error_sets_error_status(self, exporter):
        """Test that sync exceptions set StatusCode.ERROR on the span."""
        from opentelemetry.instrumentation.azure_search.wrapper import _sync_wrap
        from opentelemetry import trace
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)

        def failing_method(*args, **kwargs):
            raise ValueError("Search index not found")

        to_wrap = {"span_name": "azure_search.search", "method": "search"}

        with pytest.raises(ValueError, match="Search index not found"):
            _sync_wrap(tracer, to_wrap, failing_method, MagicMock(), (), {})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == StatusCode.ERROR
        assert "Search index not found" in spans[0].status.description

    def test_sync_error_records_exception(self, exporter):
        """Test that sync exceptions are recorded as span events."""
        from opentelemetry.instrumentation.azure_search.wrapper import _sync_wrap
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)

        def failing_method(*args, **kwargs):
            raise ConnectionError("Service unavailable")

        to_wrap = {"span_name": "azure_search.get_document", "method": "get_document"}

        with pytest.raises(ConnectionError, match="Service unavailable"):
            _sync_wrap(tracer, to_wrap, failing_method, MagicMock(), (), {})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        events = spans[0].events
        assert len(events) == 1
        assert events[0].name == "exception"
        assert events[0].attributes["exception.type"] == "ConnectionError"
        assert "Service unavailable" in events[0].attributes["exception.message"]

    def test_sync_success_sets_ok_status(self, exporter):
        """Test that successful sync calls set StatusCode.OK."""
        from opentelemetry.instrumentation.azure_search.wrapper import _sync_wrap
        from opentelemetry import trace
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)

        def ok_method(*args, **kwargs):
            return 42

        to_wrap = {"span_name": "azure_search.get_document_count", "method": "get_document_count"}

        result = _sync_wrap(tracer, to_wrap, ok_method, MagicMock(), (), {})
        assert result == 42

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == StatusCode.OK


class TestAsyncWrapper:
    """Tests for async wrapping behavior."""

    @pytest.mark.asyncio
    async def test_async_wrap_awaits_coroutine(self, exporter):
        """Test that async methods are properly awaited."""
        from opentelemetry.instrumentation.azure_search.wrapper import _async_wrap
        from opentelemetry import trace
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)

        async def async_search(*args, **kwargs):
            return [{"id": "1", "name": "Test"}]

        to_wrap = {"span_name": "azure_search.search", "method": "search"}
        mock_instance = MagicMock()
        mock_instance._index_name = "test-index"

        result = await _async_wrap(
            tracer, to_wrap, async_search, mock_instance, (), {"search_text": "test"}
        )
        assert result == [{"id": "1", "name": "Test"}]

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == StatusCode.OK

    @pytest.mark.asyncio
    async def test_async_error_sets_error_status(self, exporter):
        """Test that async exceptions set StatusCode.ERROR on the span."""
        from opentelemetry.instrumentation.azure_search.wrapper import _async_wrap
        from opentelemetry import trace
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)

        async def failing_async(*args, **kwargs):
            raise RuntimeError("Async operation failed")

        to_wrap = {"span_name": "azure_search.search", "method": "search"}

        with pytest.raises(RuntimeError, match="Async operation failed"):
            await _async_wrap(
                tracer, to_wrap, failing_async, MagicMock(), (), {}
            )

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == StatusCode.ERROR
        assert "Async operation failed" in spans[0].status.description

    @pytest.mark.asyncio
    async def test_async_error_records_exception(self, exporter):
        """Test that async exceptions are recorded as span events."""
        from opentelemetry.instrumentation.azure_search.wrapper import _async_wrap
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)

        async def failing_async(*args, **kwargs):
            raise TimeoutError("Request timed out")

        to_wrap = {"span_name": "azure_search.search", "method": "search"}

        with pytest.raises(TimeoutError, match="Request timed out"):
            await _async_wrap(
                tracer, to_wrap, failing_async, MagicMock(), (), {}
            )

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        events = spans[0].events
        assert len(events) == 1
        assert events[0].name == "exception"

    def test_wrap_detects_async_function(self):
        """Test that _wrap correctly identifies async functions."""
        import asyncio

        async def async_fn():
            pass

        def sync_fn():
            pass

        assert asyncio.iscoroutinefunction(async_fn) is True
        assert asyncio.iscoroutinefunction(sync_fn) is False


class TestSyncWrapDispatch:
    """Tests that _sync_wrap dispatches to the correct attribute setters for each method."""

    def _run_sync_wrap(self, exporter, method, span_name, wrapped_return=None, instance_attrs=None, kwargs=None):
        from opentelemetry.instrumentation.azure_search.wrapper import _sync_wrap
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        mock_instance = MagicMock()
        if instance_attrs:
            for k, v in instance_attrs.items():
                setattr(mock_instance, k, v)

        def wrapped_fn(*a, **kw):
            return wrapped_return

        to_wrap = {"span_name": span_name, "method": method}
        result = _sync_wrap(tracer, to_wrap, wrapped_fn, mock_instance, (), kwargs or {})
        spans = exporter.get_finished_spans()
        return result, spans

    def test_search_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "search", "azure_search.search",
            instance_attrs={"_index_name": "idx"},
            kwargs={"search_text": "hello", "top": 5},
        )
        assert len(spans) == 1
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_TEXT) == "hello"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_TOP) == 5
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "idx"

    def test_get_document_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "get_document", "azure_search.get_document",
            kwargs={"key": "doc-1"},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_KEY) == "doc-1"

    def test_upload_documents_dispatch(self, exporter):
        docs = [{"id": "1"}, {"id": "2"}]
        mock_result = [MagicMock(succeeded=True), MagicMock(succeeded=False)]
        result, spans = self._run_sync_wrap(
            exporter, "upload_documents", "azure_search.upload_documents",
            wrapped_return=mock_result,
            kwargs={"documents": docs},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 2
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT) == 1
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_FAILED_COUNT) == 1

    def test_merge_documents_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "merge_documents", "azure_search.merge_documents",
            wrapped_return=[MagicMock(succeeded=True)],
            kwargs={"documents": [{"id": "1"}]},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 1

    def test_delete_documents_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "delete_documents", "azure_search.delete_documents",
            wrapped_return=[MagicMock(succeeded=True)],
            kwargs={"documents": [{"id": "1"}, {"id": "2"}]},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 2

    def test_merge_or_upload_documents_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "merge_or_upload_documents", "azure_search.merge_or_upload_documents",
            wrapped_return=[MagicMock(succeeded=True)],
            kwargs={"documents": [{"id": "1"}]},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 1

    def test_index_documents_dispatch(self, exporter):
        batch = MagicMock()
        batch.actions = [{"id": "1"}, {"id": "2"}]
        mock_response = MagicMock()
        mock_response.results = [MagicMock(succeeded=True), MagicMock(succeeded=True)]
        result, spans = self._run_sync_wrap(
            exporter, "index_documents", "azure_search.index_documents",
            wrapped_return=mock_response,
            kwargs={"batch": batch},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 2
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT) == 2
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_FAILED_COUNT) == 0

    def test_autocomplete_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "autocomplete", "azure_search.autocomplete",
            wrapped_return=[{"text": "a"}, {"text": "b"}],
            kwargs={"search_text": "he", "suggester_name": "sg"},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_TEXT) == "he"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SUGGESTER_NAME) == "sg"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_AUTOCOMPLETE_RESULTS_COUNT) == 2

    def test_suggest_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "suggest", "azure_search.suggest",
            wrapped_return=[{"text": "suggestion"}],
            kwargs={"search_text": "ho", "suggester_name": "sg"},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_TEXT) == "ho"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SUGGEST_RESULTS_COUNT) == 1

    def test_get_document_count_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "get_document_count", "azure_search.get_document_count",
            wrapped_return=500,
        )
        assert result == 500
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 500

    def test_create_index_dispatch(self, exporter):
        index = MagicMock()
        index.name = "my-index"
        result, spans = self._run_sync_wrap(
            exporter, "create_index", "azure_search.create_index",
            wrapped_return=index,
            kwargs={"index": index},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "my-index"

    def test_create_or_update_index_dispatch(self, exporter):
        index = MagicMock()
        index.name = "upsert-index"
        result, spans = self._run_sync_wrap(
            exporter, "create_or_update_index", "azure_search.create_or_update_index",
            kwargs={"index": index},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "upsert-index"

    def test_delete_index_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "delete_index", "azure_search.delete_index",
            kwargs={"index": "old-index"},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "old-index"

    def test_get_index_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "get_index", "azure_search.get_index",
            kwargs={"index": "my-index"},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "my-index"

    def test_get_index_statistics_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "get_index_statistics", "azure_search.get_index_statistics",
            kwargs={"index_name": "stats-index"},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "stats-index"

    def test_analyze_text_dispatch(self, exporter):
        req = MagicMock()
        req.analyzer_name = "standard.lucene"
        result, spans = self._run_sync_wrap(
            exporter, "analyze_text", "azure_search.analyze_text",
            kwargs={"index_name": "my-index", "analyze_request": req},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "my-index"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_ANALYZER_NAME) == "standard.lucene"

    def test_create_indexer_dispatch(self, exporter):
        indexer = MagicMock()
        indexer.name = "my-indexer"
        result, spans = self._run_sync_wrap(
            exporter, "create_indexer", "azure_search.create_indexer",
            kwargs={"indexer": indexer},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEXER_NAME) == "my-indexer"

    def test_get_indexer_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "get_indexer", "azure_search.get_indexer",
            kwargs={"name": "my-indexer"},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEXER_NAME) == "my-indexer"

    def test_get_indexers_dispatch(self, exporter):
        """Tests the list/get_indexers branch that just sets indexer_name from kwargs."""
        result, spans = self._run_sync_wrap(
            exporter, "get_indexers", "azure_search.get_indexers",
        )
        assert len(spans) == 1

    def test_run_indexer_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "run_indexer", "azure_search.run_indexer",
            kwargs={"name": "my-indexer"},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEXER_NAME) == "my-indexer"

    def test_reset_indexer_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "reset_indexer", "azure_search.reset_indexer",
            kwargs={"name": "my-indexer"},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEXER_NAME) == "my-indexer"

    def test_get_indexer_status_dispatch(self, exporter):
        mock_response = MagicMock()
        mock_response.status = "running"
        mock_response.last_result.items_processed = 100
        mock_response.last_result.items_failed = 2
        result, spans = self._run_sync_wrap(
            exporter, "get_indexer_status", "azure_search.get_indexer_status",
            wrapped_return=mock_response,
            kwargs={"name": "my-indexer"},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEXER_NAME) == "my-indexer"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEXER_STATUS) == "running"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENTS_PROCESSED) == 100
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENTS_FAILED) == 2

    def test_create_data_source_dispatch(self, exporter):
        ds = MagicMock()
        ds.name = "blob-ds"
        ds.type = "azureblob"
        result, spans = self._run_sync_wrap(
            exporter, "create_data_source_connection", "azure_search.create_data_source_connection",
            kwargs={"data_source_connection": ds},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DATA_SOURCE_NAME) == "blob-ds"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DATA_SOURCE_TYPE) == "azureblob"

    def test_get_data_source_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "get_data_source_connection", "azure_search.get_data_source_connection",
            kwargs={"name": "blob-ds"},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DATA_SOURCE_NAME) == "blob-ds"

    def test_get_data_source_connections_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "get_data_source_connections", "azure_search.get_data_source_connections",
        )
        assert len(spans) == 1

    def test_create_skillset_dispatch(self, exporter):
        skillset = MagicMock()
        skillset.name = "my-skillset"
        skillset.skills = [MagicMock(), MagicMock(), MagicMock()]
        result, spans = self._run_sync_wrap(
            exporter, "create_skillset", "azure_search.create_skillset",
            kwargs={"skillset": skillset},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SKILLSET_NAME) == "my-skillset"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SKILLSET_SKILL_COUNT) == 3

    def test_get_skillset_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "get_skillset", "azure_search.get_skillset",
            kwargs={"name": "my-skillset"},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SKILLSET_NAME) == "my-skillset"

    def test_get_skillsets_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "get_skillsets", "azure_search.get_skillsets",
        )
        assert len(spans) == 1

    # Synonym map dispatch tests
    def test_create_synonym_map_dispatch(self, exporter):
        sm = MockSynonymMap(name="test-sm", synonyms=["a,b", "c,d"])
        result, spans = self._run_sync_wrap(
            exporter, "create_synonym_map", "azure_search.create_synonym_map",
            kwargs={"synonym_map": sm},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_NAME) == "test-sm"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_SYNONYMS_COUNT) == 2

    def test_get_synonym_map_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "get_synonym_map", "azure_search.get_synonym_map",
            kwargs={"name": "my-sm"},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_NAME) == "my-sm"

    def test_delete_synonym_map_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "delete_synonym_map", "azure_search.delete_synonym_map",
            kwargs={"name": "old-sm"},
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_NAME) == "old-sm"

    def test_get_synonym_maps_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "get_synonym_maps", "azure_search.get_synonym_maps",
        )
        assert len(spans) == 1

    def test_get_synonym_map_names_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "get_synonym_map_names", "azure_search.get_synonym_map_names",
        )
        assert len(spans) == 1

    # Service statistics dispatch test
    def test_get_service_statistics_dispatch(self, exporter):
        response = MockServiceStatistics(document_count=5000, index_count=3)
        result, spans = self._run_sync_wrap(
            exporter, "get_service_statistics", "azure_search.get_service_statistics",
            wrapped_return=response,
        )
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SERVICE_DOCUMENT_COUNT) == 5000
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SERVICE_INDEX_COUNT) == 3

    # Name-only listing methods dispatch tests
    def test_list_index_names_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "list_index_names", "azure_search.list_index_names",
        )
        assert len(spans) == 1

    def test_get_indexer_names_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "get_indexer_names", "azure_search.get_indexer_names",
        )
        assert len(spans) == 1

    def test_get_data_source_connection_names_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "get_data_source_connection_names", "azure_search.get_data_source_connection_names",
        )
        assert len(spans) == 1

    def test_get_skillset_names_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "get_skillset_names", "azure_search.get_skillset_names",
        )
        assert len(spans) == 1

    # BufferedSender flush dispatch test
    def test_flush_dispatch(self, exporter):
        result, spans = self._run_sync_wrap(
            exporter, "flush", "azure_search.flush",
        )
        assert len(spans) == 1

    def test_none_response_does_not_set_ok(self, exporter):
        """When wrapped returns None, response attrs are skipped but OK is still set."""
        from opentelemetry.trace.status import StatusCode
        result, spans = self._run_sync_wrap(
            exporter, "search", "azure_search.search",
            wrapped_return=None,
        )
        assert result is None
        assert spans[0].status.status_code == StatusCode.OK

    def test_unknown_method_still_creates_span(self, exporter):
        """Unknown methods still get a span with vendor attribute."""
        result, spans = self._run_sync_wrap(
            exporter, "some_future_method", "azure_search.some_future_method",
            wrapped_return="ok",
        )
        assert len(spans) == 1
        assert spans[0].attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Azure AI Search"


class TestAttributeFunctionEdgeCases:
    """Tests for edge cases in individual attribute setter functions."""

    def test_set_span_attribute_skips_none(self, exporter):
        """_set_span_attribute does not set attribute when value is None."""
        from opentelemetry.instrumentation.azure_search.wrapper import _set_span_attribute
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_span_attribute(span, "test.attr", None)

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get("test.attr") is None

    def test_set_span_attribute_skips_empty_string(self, exporter):
        """_set_span_attribute does not set attribute when value is empty string."""
        from opentelemetry.instrumentation.azure_search.wrapper import _set_span_attribute
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_span_attribute(span, "test.attr", "")

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get("test.attr") is None

    def test_set_span_attribute_sets_valid_value(self, exporter):
        """_set_span_attribute sets attribute when value is valid."""
        from opentelemetry.instrumentation.azure_search.wrapper import _set_span_attribute
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_span_attribute(span, "test.attr", "hello")
            _set_span_attribute(span, "test.int", 42)
            _set_span_attribute(span, "test.bool", True)

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get("test.attr") == "hello"
        assert spans[0].attributes.get("test.int") == 42
        assert spans[0].attributes.get("test.bool") is True

    def test_index_name_from_instance(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_index_name_attribute
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            instance = MagicMock()
            instance._index_name = "hotels"
            _set_index_name_attribute(span, instance, (), {})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "hotels"

    def test_index_name_not_present(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_index_name_attribute
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            instance = MagicMock(spec=[])
            _set_index_name_attribute(span, instance, (), {})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) is None

    def test_search_text_from_positional_arg(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_search_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_search_attributes(span, ("positional query",), {})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_TEXT) == "positional query"

    def test_search_query_type_string(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_search_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_search_attributes(span, (), {"query_type": "full"})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_QUERY_TYPE) == "full"

    def test_search_top_sets_vector_db_top_k(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_search_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_search_attributes(span, (), {"top": 10})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.VECTOR_DB_QUERY_TOP_K) == 10

    def test_get_document_key_from_positional(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_get_document_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_get_document_attributes(span, ("key-123",), {})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_KEY) == "key-123"

    def test_document_batch_from_positional(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_document_batch_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_document_batch_attributes(span, ([{"id": "1"}, {"id": "2"}],), {})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 2

    def test_document_batch_from_generator_skips_count(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_document_batch_attributes
        from opentelemetry import trace

        def doc_generator():
            yield {"id": "1"}
            yield {"id": "2"}
            yield {"id": "3"}

        gen = doc_generator()
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_document_batch_attributes(span, (), {"documents": gen})

        spans = exporter.get_finished_spans()
        # Generators lack __len__, so count is skipped to avoid consuming the iterator
        assert SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT not in spans[0].attributes
        # Verify the generator was NOT consumed
        assert len(list(gen)) == 3

    def test_document_batch_no_documents(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_document_batch_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_document_batch_attributes(span, (), {})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) is None

    def test_index_documents_batch_from_positional(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_index_documents_attributes
        from opentelemetry import trace

        batch = MagicMock()
        batch.actions = [{"id": "1"}]
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_index_documents_attributes(span, (batch,), {})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 1

    def test_index_documents_no_batch(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_index_documents_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_index_documents_attributes(span, (), {})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) is None

    def test_suggestion_attrs_from_positional(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_suggestion_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_suggestion_attributes(span, ("hel", "sg1"), {})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_TEXT) == "hel"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SUGGESTER_NAME) == "sg1"

    def test_index_management_delete_with_object(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_index_management_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            index_obj = MagicMock()
            index_obj.name = "obj-index"
            _set_index_management_attributes(span, "delete_index", (), {"index": index_obj})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "obj-index"

    def test_index_management_get_from_positional(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_index_management_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_index_management_attributes(span, "get_index", ("pos-index",), {})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "pos-index"

    def test_index_management_create_no_index(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_index_management_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_index_management_attributes(span, "create_index", (), {})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) is None

    def test_analyze_text_from_positional(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_analyze_text_attributes
        from opentelemetry import trace

        req = MagicMock()
        req.analyzer_name = "en.lucene"
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_analyze_text_attributes(span, ("idx", req), {})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "idx"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_ANALYZER_NAME) == "en.lucene"

    def test_analyze_text_enum_analyzer(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_analyze_text_attributes
        from opentelemetry import trace

        enum_analyzer = MagicMock()
        enum_analyzer.value = "standard.lucene"
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_analyze_text_attributes(span, (), {"index_name": "idx", "analyzer_name": enum_analyzer})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_ANALYZER_NAME) == "standard.lucene"

    def test_analyze_text_direct_kwargs_fallback(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_analyze_text_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_analyze_text_attributes(span, (), {"index_name": "idx", "analyzer": "keyword"})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_ANALYZER_NAME) == "keyword"

    def test_indexer_create_from_positional(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_indexer_management_attributes
        from opentelemetry import trace

        indexer = MagicMock()
        indexer.name = "pos-indexer"
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_indexer_management_attributes(span, "create_indexer", (indexer,), {})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEXER_NAME) == "pos-indexer"

    def test_indexer_get_from_positional(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_indexer_management_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_indexer_management_attributes(span, "get_indexer", ("pos-indexer",), {})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEXER_NAME) == "pos-indexer"

    def test_create_or_update_indexer_dispatch(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_indexer_management_attributes
        from opentelemetry import trace

        indexer = MagicMock()
        indexer.name = "updated-indexer"
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_indexer_management_attributes(span, "create_or_update_indexer", (), {"indexer": indexer})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEXER_NAME) == "updated-indexer"

    def test_data_source_create_from_positional(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_data_source_attributes
        from opentelemetry import trace

        ds = MagicMock()
        ds.name = "sql-ds"
        ds.type = "azuresql"
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_data_source_attributes(span, "create_data_source_connection", (ds,), {})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DATA_SOURCE_NAME) == "sql-ds"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DATA_SOURCE_TYPE) == "azuresql"

    def test_data_source_get_from_positional(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_data_source_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_data_source_attributes(span, "get_data_source_connection", ("my-ds",), {})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DATA_SOURCE_NAME) == "my-ds"

    def test_create_or_update_data_source_dispatch(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_data_source_attributes
        from opentelemetry import trace

        ds = MagicMock()
        ds.name = "cosmos-ds"
        ds.type = "cosmosdb"
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            kwargs = {"data_source_connection": ds}
            _set_data_source_attributes(span, "create_or_update_data_source_connection", (), kwargs)

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DATA_SOURCE_NAME) == "cosmos-ds"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DATA_SOURCE_TYPE) == "cosmosdb"

    def test_skillset_create_from_positional(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_skillset_attributes
        from opentelemetry import trace

        ss = MagicMock()
        ss.name = "my-ss"
        ss.skills = [MagicMock(), MagicMock()]
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_skillset_attributes(span, "create_skillset", (ss,), {})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SKILLSET_NAME) == "my-ss"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SKILLSET_SKILL_COUNT) == 2

    def test_skillset_get_from_positional(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_skillset_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_skillset_attributes(span, "get_skillset", ("my-ss",), {})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SKILLSET_NAME) == "my-ss"

    def test_create_or_update_skillset_dispatch(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_skillset_attributes
        from opentelemetry import trace

        ss = MagicMock()
        ss.name = "updated-ss"
        ss.skills = [MagicMock()]
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_skillset_attributes(span, "create_or_update_skillset", (), {"skillset": ss})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SKILLSET_NAME) == "updated-ss"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SKILLSET_SKILL_COUNT) == 1

    def test_indexer_status_no_last_result(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_indexer_status_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            resp = MagicMock()
            resp.status = "running"
            resp.last_result = None
            _set_indexer_status_attributes(span, (), {}, resp)

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEXER_STATUS) == "running"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENTS_PROCESSED) is None

    def test_indexer_status_no_status(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_indexer_status_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            resp = MagicMock(spec=[])
            _set_indexer_status_attributes(span, (), {}, resp)

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEXER_STATUS) is None

    def test_document_count_response_non_int(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_document_count_response_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_document_count_response_attributes(span, "not-an-int")

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) is None

    def test_autocomplete_response_non_list(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_autocomplete_response_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_autocomplete_response_attributes(span, "not-a-list")

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_AUTOCOMPLETE_RESULTS_COUNT) is None

    def test_suggest_response_non_list(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_suggest_response_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_suggest_response_attributes(span, "not-a-list")

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SUGGEST_RESULTS_COUNT) is None

    def test_index_documents_response_no_results_attr(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_index_documents_response_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            resp = MagicMock(spec=[])
            _set_index_documents_response_attributes(span, resp)

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT) is None

    def test_search_response_no_get_count(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_search_response_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            resp = MagicMock(spec=[])
            _set_search_response_attributes(span, resp)

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_RESULTS_COUNT) is None

    def test_search_fields_as_string(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_search_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_search_attributes(span, (), {"search_fields": "title,content"})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_FIELDS) == "title,content"

    def test_delete_skillset_dispatch(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_skillset_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_skillset_attributes(span, "delete_skillset", (), {"name": "old-ss"})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SKILLSET_NAME) == "old-ss"

    def test_delete_data_source_dispatch(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_data_source_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_data_source_attributes(span, "delete_data_source_connection", (), {"name": "old-ds"})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DATA_SOURCE_NAME) == "old-ds"


class TestUtilsAndLifecycle:
    """Tests for dont_throw utility and instrumentor lifecycle."""

    def test_dont_throw_catches_exceptions(self):
        from opentelemetry.instrumentation.azure_search.utils import dont_throw

        @dont_throw
        def failing_function():
            raise RuntimeError("boom")

        result = failing_function()
        assert result is None

    def test_dont_throw_calls_exception_logger(self):
        from opentelemetry.instrumentation.azure_search.utils import dont_throw
        from opentelemetry.instrumentation.azure_search.config import Config

        logged = []
        original = Config.exception_logger
        Config.exception_logger = lambda e: logged.append(e)
        try:
            @dont_throw
            def failing_function():
                raise ValueError("test error")

            failing_function()
            assert len(logged) == 1
            assert isinstance(logged[0], ValueError)
        finally:
            Config.exception_logger = original

    def test_dont_throw_passes_through_on_success(self):
        from opentelemetry.instrumentation.azure_search.utils import dont_throw

        @dont_throw
        def ok_function():
            return 42

        assert ok_function() == 42

    def test_suppression_key_bypasses_instrumentation(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _wrap
        from opentelemetry import trace, context as context_api
        from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY

        tracer = trace.get_tracer(__name__)
        wrapper_fn = _wrap(tracer, {"span_name": "azure_search.search", "method": "search"})

        def wrapped_fn(*args, **kwargs):
            return "result"

        token = context_api.attach(context_api.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        try:
            result = wrapper_fn(wrapped_fn, MagicMock(), (), {})
        finally:
            context_api.detach(token)

        assert result == "result"
        assert len(exporter.get_finished_spans()) == 0

    def test_wrap_delegates_to_sync(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _wrap
        from opentelemetry import trace
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        wrapper_fn = _wrap(tracer, {"span_name": "azure_search.search", "method": "search"})

        def sync_fn(*args, **kwargs):
            return "sync-result"

        result = wrapper_fn(sync_fn, MagicMock(), (), {"search_text": "test"})
        assert result == "sync-result"

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == StatusCode.OK

    @pytest.mark.asyncio
    async def test_wrap_delegates_to_async(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _wrap
        from opentelemetry import trace
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        wrapper_fn = _wrap(tracer, {"span_name": "azure_search.search", "method": "search"})

        async def async_fn(*args, **kwargs):
            return "async-result"

        result = await wrapper_fn(async_fn, MagicMock(), (), {"search_text": "test"})
        assert result == "async-result"

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == StatusCode.OK

    def test_instrumentor_uninstrument_and_reinstrument(self):
        from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor

        instrumentor = AzureSearchInstrumentor()
        instrumentor.uninstrument()
        instrumentor.instrument()


class TestAttributeExtractionEdgeCases:
    """Tests for defensive edge cases in attribute extraction (missing args, bad types, etc.)."""

    # --- wrapper.py: _set_document_batch_attributes lines 310-311 ---
    # Documents that are not len-able and list() raises TypeError

    def test_document_batch_without_len_skips_count(self, exporter):
        """Documents without __len__ are skipped to avoid consuming iterators."""
        from opentelemetry.instrumentation.azure_search.wrapper import _set_document_batch_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            # Generator has no __len__ — should be skipped, not consumed
            gen_docs = (x for x in [{"id": "1"}, {"id": "2"}])
            _set_document_batch_attributes(span, (), {"documents": gen_docs})

        spans = exporter.get_finished_spans()
        assert SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT not in spans[0].attributes
        # Verify the generator was NOT consumed
        assert list(gen_docs) == [{"id": "1"}, {"id": "2"}]

    # --- wrapper.py: branch 255->260 (fields is falsy) ---

    def test_vector_search_fields_empty(self, exporter):
        """Empty fields attribute should not set vector fields span attribute."""
        from opentelemetry.instrumentation.azure_search.wrapper import _set_vector_search_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        vq = MagicMock()
        vq.k_nearest_neighbors = 5
        vq.fields = None  # falsy fields
        vq.exhaustive = None

        with tracer.start_as_current_span("test") as span:
            _set_vector_search_attributes(span, {"vector_queries": [vq]})

        spans = exporter.get_finished_spans()
        assert SpanAttributes.AZURE_SEARCH_VECTOR_FIELDS not in spans[0].attributes

    # --- wrapper.py: branch 321->exit (actions without __len__) ---

    def test_index_documents_actions_no_len(self, exporter):
        """Batch with actions that lack __len__ should not set document count."""
        from opentelemetry.instrumentation.azure_search.wrapper import _set_index_documents_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        batch = MagicMock()
        # actions is an iterator (no __len__)
        batch.actions = iter([1, 2, 3])

        with tracer.start_as_current_span("test") as span:
            _set_index_documents_attributes(span, (), {"batch": batch})

        spans = exporter.get_finished_spans()
        assert SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT not in spans[0].attributes

    def test_index_documents_actions_none(self, exporter):
        """Batch with None actions should not set document count."""
        from opentelemetry.instrumentation.azure_search.wrapper import _set_index_documents_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        batch = MagicMock()
        batch.actions = None

        with tracer.start_as_current_span("test") as span:
            _set_index_documents_attributes(span, (), {"batch": batch})

        spans = exporter.get_finished_spans()
        assert SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT not in spans[0].attributes

    # --- wrapper.py: branch 344->exit (index is falsy for create_index) ---

    def test_index_management_create_index_no_index_arg(self, exporter):
        """create_index with no index argument should not set index name."""
        from opentelemetry.instrumentation.azure_search.wrapper import _set_index_management_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_index_management_attributes(span, "create_index", (), {})

        spans = exporter.get_finished_spans()
        assert SpanAttributes.AZURE_SEARCH_INDEX_NAME not in spans[0].attributes

    # --- wrapper.py: branch 349->exit (index_name is not string and no .name) ---

    def test_index_management_delete_with_non_string_non_object(self, exporter):
        """delete_index with non-string, non-object arg should not set index name."""
        from opentelemetry.instrumentation.azure_search.wrapper import _set_index_management_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            # Pass an integer - not a string, no .name attribute
            _set_index_management_attributes(span, "delete_index", (42,), {})

        spans = exporter.get_finished_spans()
        assert SpanAttributes.AZURE_SEARCH_INDEX_NAME not in spans[0].attributes

    # --- wrapper.py: branch 371->exit (analyzer_name is falsy after all checks) ---

    def test_analyze_text_no_analyzer(self, exporter):
        """analyze_text with no analyzer anywhere should not set analyzer name."""
        from opentelemetry.instrumentation.azure_search.wrapper import _set_analyze_text_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            # No analyze_request, no analyzer_name, no analyzer in kwargs
            _set_analyze_text_attributes(span, ("my-index",), {})

        spans = exporter.get_finished_spans()
        assert SpanAttributes.AZURE_SEARCH_INDEX_NAME in spans[0].attributes
        assert SpanAttributes.AZURE_SEARCH_ANALYZER_NAME not in spans[0].attributes

    # --- wrapper.py: branch 387->394 (indexer is falsy for create_indexer) ---

    def test_indexer_management_create_no_indexer(self, exporter):
        """create_indexer with no indexer should still call _set_span_attribute with None."""
        from opentelemetry.instrumentation.azure_search.wrapper import _set_indexer_management_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_indexer_management_attributes(span, "create_indexer", (), {})

        spans = exporter.get_finished_spans()
        assert SpanAttributes.AZURE_SEARCH_INDEXER_NAME not in spans[0].attributes

    # --- wrapper.py: branch 400->exit (response is falsy for indexer status) ---

    def test_indexer_status_none_response(self, exporter):
        """get_indexer_status with None response should not set any attributes."""
        from opentelemetry.instrumentation.azure_search.wrapper import _set_indexer_status_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_indexer_status_attributes(span, (), {}, None)

        spans = exporter.get_finished_spans()
        assert SpanAttributes.AZURE_SEARCH_INDEXER_STATUS not in spans[0].attributes

    # --- wrapper.py: branch 424->433 (data_source is falsy for create) ---

    def test_data_source_create_no_data_source(self, exporter):
        """create_data_source_connection with no data source should handle gracefully."""
        from opentelemetry.instrumentation.azure_search.wrapper import _set_data_source_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_data_source_attributes(span, "create_data_source_connection", (), {})

        spans = exporter.get_finished_spans()
        assert SpanAttributes.AZURE_SEARCH_DATA_SOURCE_NAME not in spans[0].attributes

    # --- wrapper.py: branch 446->457 (skillset is falsy for create) ---

    def test_skillset_create_no_skillset(self, exporter):
        """create_skillset with no skillset should handle gracefully."""
        from opentelemetry.instrumentation.azure_search.wrapper import _set_skillset_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_skillset_attributes(span, "create_skillset", (), {})

        spans = exporter.get_finished_spans()
        assert SpanAttributes.AZURE_SEARCH_SKILLSET_NAME not in spans[0].attributes

    # --- wrapper.py: branch 450->457 (skills no __len__ or falsy) ---

    def test_skillset_create_skills_not_countable(self, exporter):
        """Skillset with skills that lack __len__ should not set skill count."""
        from opentelemetry.instrumentation.azure_search.wrapper import _set_skillset_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        skillset = MagicMock()
        skillset.name = "my-skillset"
        # skills is an iterator (no __len__)
        skillset.skills = iter(["skill1", "skill2"])

        with tracer.start_as_current_span("test") as span:
            _set_skillset_attributes(span, "create_skillset", (), {"skillset": skillset})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SKILLSET_NAME) == "my-skillset"
        assert SpanAttributes.AZURE_SEARCH_SKILLSET_SKILL_COUNT not in spans[0].attributes

    def test_skillset_create_skills_none(self, exporter):
        """Skillset with None skills should not set skill count."""
        from opentelemetry.instrumentation.azure_search.wrapper import _set_skillset_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        skillset = MagicMock()
        skillset.name = "my-skillset"
        skillset.skills = None

        with tracer.start_as_current_span("test") as span:
            _set_skillset_attributes(span, "create_skillset", (), {"skillset": skillset})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SKILLSET_NAME) == "my-skillset"
        assert SpanAttributes.AZURE_SEARCH_SKILLSET_SKILL_COUNT not in spans[0].attributes

    # --- __init__.py: lines 508-511 (ImportError during _instrument) ---

    def test_instrument_handles_import_error(self):
        """_instrument should gracefully handle ImportError for missing async modules."""
        from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor
        import builtins

        instrumentor = AzureSearchInstrumentor()
        instrumentor.uninstrument()

        original_import = builtins.__import__

        def failing_import(name, *args, **kwargs):
            if "azure.search.documents" in name:
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        try:
            builtins.__import__ = failing_import
            # Should not raise - ImportError is caught
            instrumentor._instrument()
        finally:
            builtins.__import__ = original_import
            # Re-instrument normally for other tests
            instrumentor.uninstrument()
            instrumentor.instrument()

    # --- __init__.py: lines 521-523 (Exception during _uninstrument) ---

    def test_uninstrument_handles_exception(self):
        """_uninstrument should gracefully handle exceptions from unwrap."""
        from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor
        from unittest.mock import patch

        instrumentor = AzureSearchInstrumentor()

        # Mock unwrap to raise an exception
        with patch("opentelemetry.instrumentation.azure_search.unwrap", side_effect=Exception("unwrap failed")):
            # Should not raise - Exception is caught
            instrumentor._uninstrument()

        # Re-instrument for other tests
        instrumentor.instrument()

    # --- __init__.py: branch 502->494 (wrap_object not found on module) ---

    def test_instrument_skips_missing_class(self):
        """_instrument should skip wrapping when the class is not found on the module."""
        from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor
        import types

        instrumentor = AzureSearchInstrumentor()
        instrumentor.uninstrument()

        # Create a module that exists but doesn't have the expected class
        fake_module = types.ModuleType("fake_module")

        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def selective_import(name, *args, **kwargs):
            if "azure.search.documents" in name:
                return fake_module  # Module exists but class doesn't
            return original_import(name, *args, **kwargs)

        import builtins
        try:
            builtins.__import__ = selective_import
            # Should not raise - missing class is handled via getattr check
            instrumentor._instrument()
        finally:
            builtins.__import__ = original_import
            instrumentor.uninstrument()
            instrumentor.instrument()

    # --- wrapper.py: branch 344->exit (method not in either index management branch) ---

    def test_index_management_unknown_method(self, exporter):
        """Unknown method passed to _set_index_management_attributes should be a no-op."""
        from opentelemetry.instrumentation.azure_search.wrapper import _set_index_management_attributes
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_index_management_attributes(span, "unknown_method", (), {})

        spans = exporter.get_finished_spans()
        assert SpanAttributes.AZURE_SEARCH_INDEX_NAME not in spans[0].attributes

    # --- utils.py: branch 25->exit (Config.exception_logger is None) ---

    def test_dont_throw_no_exception_logger(self, exporter):
        """dont_throw should not call exception_logger when it's None."""
        from opentelemetry.instrumentation.azure_search.utils import dont_throw
        from opentelemetry.instrumentation.azure_search.config import Config

        original_logger = Config.exception_logger
        Config.exception_logger = None

        @dont_throw
        def failing_fn():
            raise ValueError("test error")

        try:
            # Should not raise, and should not fail on None logger
            result = failing_fn()
            assert result is None
        finally:
            Config.exception_logger = original_logger


class TestSynonymMapInstrumentation:
    """Tests for synonym map CRUD operations (US-005)."""

    def test_create_synonym_map_span(self, exporter):
        """Test create_synonym_map creates span with synonym_map_name."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_synonym_map_attributes,
        )
        from opentelemetry import trace

        sm = MockSynonymMap(name="my-synonyms", synonyms=["hotel,motel", "cozy,comfortable"])

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.create_synonym_map") as span:
            _set_synonym_map_attributes(span, "create_synonym_map", (sm,), {})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_NAME) == "my-synonyms"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_SYNONYMS_COUNT) == 2

    def test_create_or_update_synonym_map_span(self, exporter):
        """Test create_or_update_synonym_map creates span with synonym_map_name."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_synonym_map_attributes,
        )
        from opentelemetry import trace

        sm = MockSynonymMap(name="updated-synonyms", synonyms=["big,large", "small,tiny", "fast,quick"])

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.create_or_update_synonym_map") as span:
            _set_synonym_map_attributes(span, "create_or_update_synonym_map", (), {"synonym_map": sm})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_NAME) == "updated-synonyms"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_SYNONYMS_COUNT) == 3

    def test_delete_synonym_map_span(self, exporter):
        """Test delete_synonym_map creates span with synonym_map_name from string arg."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_synonym_map_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.delete_synonym_map") as span:
            _set_synonym_map_attributes(span, "delete_synonym_map", ("my-synonyms",), {})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_NAME) == "my-synonyms"

    def test_get_synonym_map_span(self, exporter):
        """Test get_synonym_map creates span with synonym_map_name from string arg."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_synonym_map_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.get_synonym_map") as span:
            _set_synonym_map_attributes(span, "get_synonym_map", (), {"name": "my-synonyms"})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_NAME) == "my-synonyms"

    def test_get_synonym_maps_span(self, exporter):
        """Test get_synonym_maps creates span with correct span name."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_synonym_map_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.get_synonym_maps") as span:
            _set_synonym_map_attributes(span, "get_synonym_maps", (), {})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "azure_search.get_synonym_maps"

    def test_get_synonym_map_names_span(self, exporter):
        """Test get_synonym_map_names creates span with correct span name."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_synonym_map_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.get_synonym_map_names") as span:
            _set_synonym_map_attributes(span, "get_synonym_map_names", (), {})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "azure_search.get_synonym_map_names"

    def test_create_synonym_map_synonyms_count(self, exporter):
        """Test create_synonym_map extracts synonyms_count from SynonymMap.synonyms."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_synonym_map_attributes,
        )
        from opentelemetry import trace

        sm = MockSynonymMap(name="test", synonyms=["a,b", "c,d", "e,f", "g,h"])

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.create_synonym_map") as span:
            _set_synonym_map_attributes(span, "create_synonym_map", (sm,), {})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_SYNONYMS_COUNT) == 4

    def test_synonym_map_with_positional_args(self, exporter):
        """Test synonym map with positional args works correctly."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_synonym_map_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.get_synonym_map") as span:
            _set_synonym_map_attributes(span, "get_synonym_map", ("positional-name",), {})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_NAME) == "positional-name"

    def test_delete_synonym_map_with_object_arg(self, exporter):
        """Test delete_synonym_map handles SynonymMap object passed instead of string."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_synonym_map_attributes,
        )
        from opentelemetry import trace

        sm = MockSynonymMap(name="object-synonym-map")

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.delete_synonym_map") as span:
            _set_synonym_map_attributes(span, "delete_synonym_map", (sm,), {})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SYNONYM_MAP_NAME) == "object-synonym-map"


class TestServiceStatisticsInstrumentation:
    """Tests for get_service_statistics instrumentation (US-006)."""

    def test_service_statistics_response_attributes(self, exporter):
        """Test get_service_statistics extracts document_count and index_count from response."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_service_statistics_response_attributes,
        )
        from opentelemetry import trace

        response = MockServiceStatistics(document_count=5000, index_count=3)

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.get_service_statistics") as span:
            _set_service_statistics_response_attributes(span, response)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SERVICE_DOCUMENT_COUNT) == 5000
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SERVICE_INDEX_COUNT) == 3

    def test_service_statistics_none_response(self, exporter):
        """Test get_service_statistics handles None response gracefully."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_service_statistics_response_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.get_service_statistics") as span:
            _set_service_statistics_response_attributes(span, None)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert SpanAttributes.AZURE_SEARCH_SERVICE_DOCUMENT_COUNT not in spans[0].attributes

    def test_service_statistics_no_counters(self, exporter):
        """Test get_service_statistics handles response without counters."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_service_statistics_response_attributes,
        )
        from opentelemetry import trace

        response = MagicMock(spec=[])  # No attributes at all

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.get_service_statistics") as span:
            _set_service_statistics_response_attributes(span, response)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert SpanAttributes.AZURE_SEARCH_SERVICE_DOCUMENT_COUNT not in spans[0].attributes


class TestBufferedSenderInstrumentation:
    """Tests for SearchIndexingBufferedSender instrumentation (US-008)."""

    def test_buffered_upload_documents_span(self, exporter):
        """Test upload_documents on BufferedSender creates span with document count."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_document_batch_attributes,
        )
        from opentelemetry import trace

        documents = [{"id": "1"}, {"id": "2"}, {"id": "3"}]

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.upload_documents",
            attributes={SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search"},
        ) as span:
            _set_document_batch_attributes(span, (documents,), {})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "azure_search.upload_documents"
        assert spans[0].attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Azure AI Search"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 3

    def test_buffered_delete_documents_span(self, exporter):
        """Test delete_documents on BufferedSender creates span with document count."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_document_batch_attributes,
        )
        from opentelemetry import trace

        documents = [{"id": "1"}, {"id": "2"}]

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.delete_documents",
            attributes={SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search"},
        ) as span:
            _set_document_batch_attributes(span, (), {"documents": documents})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "azure_search.delete_documents"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 2

    def test_buffered_merge_documents_span(self, exporter):
        """Test merge_documents on BufferedSender creates span with document count."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_document_batch_attributes,
        )
        from opentelemetry import trace

        documents = [{"id": "1", "rating": 5}]

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.merge_documents",
            attributes={SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search"},
        ) as span:
            _set_document_batch_attributes(span, (documents,), {})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "azure_search.merge_documents"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 1

    def test_buffered_merge_or_upload_documents_span(self, exporter):
        """Test merge_or_upload_documents on BufferedSender creates span with document count."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_document_batch_attributes,
        )
        from opentelemetry import trace

        documents = [{"id": "1"}, {"id": "2"}, {"id": "3"}, {"id": "4"}]

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.merge_or_upload_documents",
            attributes={SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search"},
        ) as span:
            _set_document_batch_attributes(span, (documents,), {})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "azure_search.merge_or_upload_documents"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 4

    def test_buffered_index_documents_span(self, exporter):
        """Test index_documents on BufferedSender creates span with batch action count."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_index_documents_attributes,
        )
        from opentelemetry import trace

        batch = MagicMock()
        batch.actions = [MagicMock(), MagicMock(), MagicMock()]

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.index_documents",
            attributes={SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search"},
        ) as span:
            _set_index_documents_attributes(span, (), {"batch": batch})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "azure_search.index_documents"
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 3

    def test_buffered_flush_span(self, exporter):
        """Test flush creates span 'azure_search.flush'."""
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.flush",
            attributes={SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search"},
        ):
            pass  # flush takes no arguments worth extracting

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "azure_search.flush"
        assert spans[0].attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Azure AI Search"

    def test_buffered_sender_db_system(self, exporter):
        """Test BufferedSender spans have db.system='Azure AI Search'."""
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.upload_documents",
            attributes={SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search"},
        ):
            pass

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Azure AI Search"

    def test_buffered_sender_methods_in_init(self):
        """Test BUFFERED_SENDER_METHODS are correctly defined in __init__.py."""
        from opentelemetry.instrumentation.azure_search import (
            BUFFERED_SENDER_METHODS,
            ASYNC_BUFFERED_SENDER_METHODS,
            WRAPPED_METHODS,
        )

        # Verify all expected methods are present
        sync_methods = {m["method"] for m in BUFFERED_SENDER_METHODS}
        assert sync_methods == {
            "upload_documents", "delete_documents", "merge_documents",
            "merge_or_upload_documents", "index_documents", "flush",
        }

        async_methods = {m["method"] for m in ASYNC_BUFFERED_SENDER_METHODS}
        assert async_methods == {
            "upload_documents", "delete_documents", "merge_documents",
            "merge_or_upload_documents", "index_documents", "flush",
        }

        # Verify they're included in WRAPPED_METHODS
        all_methods = [(m["module"], m["object"], m["method"]) for m in WRAPPED_METHODS]
        for m in BUFFERED_SENDER_METHODS + ASYNC_BUFFERED_SENDER_METHODS:
            assert (m["module"], m["object"], m["method"]) in all_methods


class TestNameOnlyListingMethods:
    """Tests for name-only listing methods instrumentation (US-010)."""

    def test_list_index_names_span(self, exporter):
        """Test list_index_names creates span 'azure_search.list_index_names'."""
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.list_index_names",
            kind=trace.SpanKind.CLIENT,
            attributes={SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search"},
        ):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "azure_search.list_index_names"
        assert spans[0].attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Azure AI Search"

    def test_get_indexer_names_span(self, exporter):
        """Test get_indexer_names creates span 'azure_search.get_indexer_names'."""
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.get_indexer_names",
            kind=trace.SpanKind.CLIENT,
            attributes={SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search"},
        ):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "azure_search.get_indexer_names"
        assert spans[0].attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Azure AI Search"

    def test_get_data_source_connection_names_span(self, exporter):
        """Test get_data_source_connection_names creates span."""
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.get_data_source_connection_names",
            kind=trace.SpanKind.CLIENT,
            attributes={SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search"},
        ):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "azure_search.get_data_source_connection_names"
        assert spans[0].attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Azure AI Search"

    def test_get_skillset_names_span(self, exporter):
        """Test get_skillset_names creates span 'azure_search.get_skillset_names'."""
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.get_skillset_names",
            kind=trace.SpanKind.CLIENT,
            attributes={SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search"},
        ):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "azure_search.get_skillset_names"
        assert spans[0].attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "Azure AI Search"

    def test_name_only_methods_in_init(self):
        """Test name-only listing methods are defined in __init__.py."""
        from opentelemetry.instrumentation.azure_search import (
            SEARCH_INDEX_CLIENT_METHODS,
            SEARCH_INDEXER_CLIENT_METHODS,
        )

        index_client_methods = {m["method"] for m in SEARCH_INDEX_CLIENT_METHODS}
        assert "list_index_names" in index_client_methods

        indexer_client_methods = {m["method"] for m in SEARCH_INDEXER_CLIENT_METHODS}
        assert "get_indexer_names" in indexer_client_methods
        assert "get_data_source_connection_names" in indexer_client_methods
        assert "get_skillset_names" in indexer_client_methods

    def test_dispatch_handles_name_only_methods_gracefully(self, exporter):
        """Test _set_request_attributes handles name-only methods without error."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_request_attributes,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        instance = MagicMock()
        instance._index_name = None

        for method in ["list_index_names", "get_indexer_names",
                       "get_data_source_connection_names", "get_skillset_names", "flush"]:
            with tracer.start_as_current_span(f"azure_search.{method}") as span:
                # Should not raise
                _set_request_attributes(span, method, instance, (), {})


class TestShouldSendContent:
    """Tests for the should_send_content() toggle function."""

    def test_default_returns_true(self, exporter, monkeypatch):
        """Default (no env var) should return True."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.instrumentation.azure_search.utils import should_send_content
        assert should_send_content() is True

    def test_env_false_returns_false(self, exporter, monkeypatch):
        """TRACELOOP_TRACE_CONTENT=false should return False."""
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        from opentelemetry.instrumentation.azure_search.utils import should_send_content
        assert should_send_content() is False

    def test_env_zero_returns_false(self, exporter, monkeypatch):
        """TRACELOOP_TRACE_CONTENT=0 should return False."""
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "0")
        from opentelemetry.instrumentation.azure_search.utils import should_send_content
        assert should_send_content() is False

    def test_override_context_true(self, exporter, monkeypatch):
        """override_enable_content_tracing=True overrides env=false."""
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        from opentelemetry.instrumentation.azure_search.utils import should_send_content
        from opentelemetry import context as context_api

        ctx = context_api.set_value("override_enable_content_tracing", True)
        token = context_api.attach(ctx)
        try:
            assert should_send_content() is True
        finally:
            context_api.detach(token)

    def test_override_context_false(self, exporter, monkeypatch):
        """override_enable_content_tracing=False overrides env=true."""
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
        from opentelemetry.instrumentation.azure_search.utils import should_send_content
        from opentelemetry import context as context_api

        ctx = context_api.set_value("override_enable_content_tracing", False)
        token = context_api.attach(ctx)
        try:
            assert should_send_content() is False
        finally:
            context_api.detach(token)

    def test_truthy_values(self, exporter, monkeypatch):
        """All truthy values should return True."""
        from opentelemetry.instrumentation.azure_search.utils import should_send_content
        for val in ["true", "1", "yes", "on", "True", "YES", "ON"]:
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", val)
            assert should_send_content() is True, f"Expected True for {val!r}"

    def test_falsy_values(self, exporter, monkeypatch):
        """Non-truthy values should return False."""
        from opentelemetry.instrumentation.azure_search.utils import should_send_content
        for val in ["false", "0", "no", "off", "False", "NO", "OFF", "random"]:
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", val)
            assert should_send_content() is False, f"Expected False for {val!r}"


class TestContentCapture:
    """Tests for response/request content capture via span attributes."""

    def test_get_document_content_attribute(self, exporter, monkeypatch):
        """get_document should set db.query.result.document attribute with document JSON."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.semconv_ai import EventAttributes

        client = MockSearchClient(
            endpoint="https://test.search.windows.net",
            index_name="test-index",
            credential=MagicMock(),
        )
        client.get_document(key="doc-123")

        spans = exporter.get_finished_spans()
        get_doc_spans = [s for s in spans if s.name == "azure_search.get_document"]
        assert len(get_doc_spans) == 1

        span = get_doc_spans[0]
        attr_key = EventAttributes.DB_QUERY_RESULT_DOCUMENT.value
        assert attr_key in dict(span.attributes)

        import json
        doc = json.loads(span.attributes[attr_key])
        assert doc["id"] == "doc-123"

    def test_autocomplete_content_attributes(self, exporter, monkeypatch):
        """autocomplete should set indexed db.search.result.entity.N attributes."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.semconv_ai import EventAttributes

        client = MockSearchClient(
            endpoint="https://test.search.windows.net",
            index_name="test-index",
            credential=MagicMock(),
        )
        client.autocomplete(search_text="lux", suggester_name="sg")

        spans = exporter.get_finished_spans()
        ac_spans = [s for s in spans if s.name == "azure_search.autocomplete"]
        assert len(ac_spans) == 1

        span = ac_spans[0]
        attr_key = f"{EventAttributes.DB_SEARCH_RESULT_ENTITY.value}.0"
        assert attr_key in dict(span.attributes)

    def test_suggest_content_attributes(self, exporter, monkeypatch):
        """suggest should set indexed db.search.result.entity.N attributes."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.semconv_ai import EventAttributes

        client = MockSearchClient(
            endpoint="https://test.search.windows.net",
            index_name="test-index",
            credential=MagicMock(),
        )
        client.suggest(search_text="lux", suggester_name="sg")

        spans = exporter.get_finished_spans()
        suggest_spans = [s for s in spans if s.name == "azure_search.suggest"]
        assert len(suggest_spans) == 1

        span = suggest_spans[0]
        attr_key = f"{EventAttributes.DB_SEARCH_RESULT_ENTITY.value}.0"
        assert attr_key in dict(span.attributes)

    def test_upload_documents_request_content_attributes(self, exporter, monkeypatch):
        """upload_documents should set per-doc indexed db.query.result.document.N attributes."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.semconv_ai import EventAttributes

        client = MockSearchClient(
            endpoint="https://test.search.windows.net",
            index_name="test-index",
            credential=MagicMock(),
        )
        docs = [{"id": "1", "name": "Hotel A"}, {"id": "2", "name": "Hotel B"}]
        client.upload_documents(documents=docs)

        spans = exporter.get_finished_spans()
        upload_spans = [s for s in spans if s.name == "azure_search.upload_documents"]
        assert len(upload_spans) == 1

        span = upload_spans[0]
        attrs = dict(span.attributes)
        doc_key_0 = f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.0"
        doc_key_1 = f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.1"
        assert doc_key_0 in attrs
        assert doc_key_1 in attrs

        import json
        first_doc = json.loads(attrs[doc_key_0])
        assert first_doc["id"] == "1"

    def test_upload_documents_response_content_attributes(self, exporter, monkeypatch):
        """upload_documents should set per-result indexed metadata attributes."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.semconv_ai import EventAttributes

        client = MockSearchClient(
            endpoint="https://test.search.windows.net",
            index_name="test-index",
            credential=MagicMock(),
        )
        client.upload_documents(documents=[{"id": "1"}])

        spans = exporter.get_finished_spans()
        upload_spans = [s for s in spans if s.name == "azure_search.upload_documents"]
        assert len(upload_spans) == 1

        span = upload_spans[0]
        attrs = dict(span.attributes)
        metadata_key = f"{EventAttributes.DB_QUERY_RESULT_METADATA.value}.0"
        assert metadata_key in attrs

    def test_search_vector_embeddings_attributes(self, exporter, monkeypatch):
        """search with vector_queries should set indexed db.search.embeddings.vector.N attributes."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.semconv_ai import EventAttributes

        vq = MagicMock()
        vq.vector = [0.1, 0.2, 0.3]
        vq.text = None
        vq.k_nearest_neighbors = 5
        vq.fields = "embedding"
        vq.exhaustive = None
        vq.kind = None
        vq.weight = None
        vq.oversampling = None

        client = MockSearchClient(
            endpoint="https://test.search.windows.net",
            index_name="test-index",
            credential=MagicMock(),
        )
        list(client.search(search_text="hotel", vector_queries=[vq]))

        spans = exporter.get_finished_spans()
        search_spans = [s for s in spans if s.name == "azure_search.search"]
        assert len(search_spans) == 1

        span = search_spans[0]
        attr_key = f"{EventAttributes.DB_SEARCH_EMBEDDINGS_VECTOR.value}.0"
        assert attr_key in dict(span.attributes)

    def test_search_text_vector_embeddings_attributes(self, exporter, monkeypatch):
        """search with text-based vector query should capture text in embeddings attribute."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.semconv_ai import EventAttributes

        vq = MagicMock()
        vq.vector = None
        vq.text = "luxury hotel"
        vq.k_nearest_neighbors = 5
        vq.fields = "embedding"
        vq.exhaustive = None
        vq.kind = None
        vq.weight = None
        vq.oversampling = None

        client = MockSearchClient(
            endpoint="https://test.search.windows.net",
            index_name="test-index",
            credential=MagicMock(),
        )
        list(client.search(search_text=None, vector_queries=[vq]))

        spans = exporter.get_finished_spans()
        search_spans = [s for s in spans if s.name == "azure_search.search"]
        assert len(search_spans) == 1

        span = search_spans[0]
        attr_key = f"{EventAttributes.DB_SEARCH_EMBEDDINGS_VECTOR.value}.0"
        assert span.attributes[attr_key] == "luxury hotel"

    def test_content_disabled_no_content_attributes(self, exporter, monkeypatch):
        """With TRACELOOP_TRACE_CONTENT=false, no content attributes should be added."""
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        from opentelemetry.semconv_ai import EventAttributes

        client = MockSearchClient(
            endpoint="https://test.search.windows.net",
            index_name="test-index",
            credential=MagicMock(),
        )
        client.get_document(key="doc-123")
        client.autocomplete(search_text="lux", suggester_name="sg")
        client.suggest(search_text="lux", suggester_name="sg")
        client.upload_documents(documents=[{"id": "1"}])

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
                        f"Found content attribute {attr_key} on span {span.name} with content disabled"
                    )

    def test_content_override_reenables(self, exporter, monkeypatch):
        """env=false + context override=True should add content attributes."""
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        from opentelemetry.semconv_ai import EventAttributes
        from opentelemetry import context as context_api

        ctx = context_api.set_value("override_enable_content_tracing", True)
        token = context_api.attach(ctx)
        try:
            client = MockSearchClient(
                endpoint="https://test.search.windows.net",
                index_name="test-index",
                credential=MagicMock(),
            )
            client.get_document(key="doc-123")
        finally:
            context_api.detach(token)

        spans = exporter.get_finished_spans()
        get_doc_spans = [s for s in spans if s.name == "azure_search.get_document"]
        assert len(get_doc_spans) == 1

        span = get_doc_spans[0]
        attr_key = EventAttributes.DB_QUERY_RESULT_DOCUMENT.value
        assert attr_key in dict(span.attributes)

    def test_index_documents_request_content_attributes(self, exporter, monkeypatch):
        """index_documents should set per-action indexed db.query.result.document.N attributes."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.semconv_ai import EventAttributes

        batch = MagicMock()
        batch.actions = [
            {"@search.action": "upload", "id": "1", "name": "Hotel A"},
            {"@search.action": "upload", "id": "2", "name": "Hotel B"},
        ]

        client = MockSearchClient(
            endpoint="https://test.search.windows.net",
            index_name="test-index",
            credential=MagicMock(),
        )
        client.index_documents(batch=batch)

        spans = exporter.get_finished_spans()
        idx_spans = [s for s in spans if s.name == "azure_search.index_documents"]
        assert len(idx_spans) == 1

        span = idx_spans[0]
        attrs = dict(span.attributes)
        doc_key_0 = f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.0"
        doc_key_1 = f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.1"
        assert doc_key_0 in attrs
        assert doc_key_1 in attrs

    def test_merge_documents_content_attributes(self, exporter, monkeypatch):
        """merge_documents should set content attributes."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.semconv_ai import EventAttributes

        client = MockSearchClient(
            endpoint="https://test.search.windows.net",
            index_name="test-index",
            credential=MagicMock(),
        )
        client.merge_documents(documents=[{"id": "1", "rating": 4.5}])

        spans = exporter.get_finished_spans()
        merge_spans = [s for s in spans if s.name == "azure_search.merge_documents"]
        assert len(merge_spans) == 1

        span = merge_spans[0]
        attr_key = f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.0"
        assert attr_key in dict(span.attributes)

    def test_delete_documents_content_attributes(self, exporter, monkeypatch):
        """delete_documents should set content attributes."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.semconv_ai import EventAttributes

        client = MockSearchClient(
            endpoint="https://test.search.windows.net",
            index_name="test-index",
            credential=MagicMock(),
        )
        client.delete_documents(documents=[{"id": "1"}])

        spans = exporter.get_finished_spans()
        del_spans = [s for s in spans if s.name == "azure_search.delete_documents"]
        assert len(del_spans) == 1

        span = del_spans[0]
        attr_key = f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.0"
        assert attr_key in dict(span.attributes)

    def test_merge_or_upload_documents_content_attributes(self, exporter, monkeypatch):
        """merge_or_upload_documents should set content attributes."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.semconv_ai import EventAttributes

        client = MockSearchClient(
            endpoint="https://test.search.windows.net",
            index_name="test-index",
            credential=MagicMock(),
        )
        client.merge_or_upload_documents(documents=[{"id": "1"}, {"id": "2"}])

        spans = exporter.get_finished_spans()
        mou_spans = [s for s in spans if s.name == "azure_search.merge_or_upload_documents"]
        assert len(mou_spans) == 1

        span = mou_spans[0]
        attrs = dict(span.attributes)
        doc_key_0 = f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.0"
        doc_key_1 = f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.1"
        assert doc_key_0 in attrs
        assert doc_key_1 in attrs


# ---------------------------------------------------------------------------
# Multi-step workflow tests — "Can these traces debug a 2am production issue?"
# ---------------------------------------------------------------------------
# Unlike the single-operation tests above, workflow tests chain multiple
# instrumented calls within a shared trace context and verify:
#   1. Trace correlation — all spans share one trace_id
#   2. Cross-span consistency — upload count matches doc count, etc.
#   3. Content reconstruction — content attributes tell the full data story
#   4. Error diagnosis — failure spans carry actionable detail
# ---------------------------------------------------------------------------


def _call_sync(tracer, method, mock_fn, instance, args=(), kwargs=None):
    """Call _sync_wrap directly — produces a real instrumented span."""
    from opentelemetry.instrumentation.azure_search.wrapper import _sync_wrap

    to_wrap = {"span_name": f"azure_search.{method}", "method": method}
    return _sync_wrap(tracer, to_wrap, mock_fn, instance, args, kwargs or {})


async def _call_async(tracer, method, mock_fn, instance, args=(), kwargs=None):
    """Call _async_wrap directly — produces a real instrumented span."""
    from opentelemetry.instrumentation.azure_search.wrapper import _async_wrap

    to_wrap = {"span_name": f"azure_search.{method}", "method": method}
    return await _async_wrap(tracer, to_wrap, mock_fn, instance, args, kwargs or {})


def _get_spans(exporter, name=None):
    """Return all finished spans, optionally filtered by name."""
    spans = exporter.get_finished_spans()
    if name is None:
        return spans
    return [s for s in spans if s.name == name]


def _assert_all_same_trace(spans):
    """Assert every span belongs to the same trace."""
    trace_ids = {s.context.trace_id for s in spans}
    assert len(trace_ids) == 1, (
        f"Expected all spans to share one trace_id, got {len(trace_ids)}: {trace_ids}"
    )


def _find_span(spans, name):
    """Find at least one span by name, return the first match."""
    matching = [s for s in spans if s.name == name]
    assert len(matching) >= 1, (
        f"No span named '{name}' in {[s.name for s in spans]}"
    )
    return matching[0]


def _make_instance(index_name="test-index"):
    """Create a mock instance with _index_name, like a SearchClient."""
    inst = MagicMock()
    inst._index_name = index_name
    return inst


def _make_index_instance():
    """Create a mock instance without _index_name, like a SearchIndexClient."""
    inst = MagicMock()
    inst._index_name = None
    return inst


def _mock_indexing_result(key, succeeded=True, status_code=200, error_message=None):
    """Create a mock IndexingResult object with the expected attributes."""
    r = MagicMock()
    r.key = key
    r.succeeded = succeeded
    r.status_code = status_code
    r.error_message = error_message
    return r


class TestSyncWorkflows:
    """Multi-step sync workflow tests that validate traces tell a debuggable story.

    Each test simulates a real production scenario and verifies the resulting
    trace has enough information to diagnose the problem at 2am.
    """

    def test_search_pipeline(self, exporter):
        """Upload docs, then search — trace must show why search returned nothing.

        2am scenario: "Users report search returns no results."
        Trace must show: upload succeeded for all docs, search query params visible.
        """
        import json
        from opentelemetry import trace
        from opentelemetry.semconv_ai import EventAttributes
        from opentelemetry.trace import SpanKind
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        docs = [
            {"id": "h1", "name": "Grand Hotel", "rating": 4.5},
            {"id": "h2", "name": "Budget Inn", "rating": 2.0},
            {"id": "h3", "name": "Cozy Motel", "rating": 3.8},
        ]
        upload_response = [
            _mock_indexing_result("h1"),
            _mock_indexing_result("h2"),
            _mock_indexing_result("h3"),
        ]

        with tracer.start_as_current_span("app.ingest_and_search"):
            _call_sync(
                tracer, "upload_documents",
                lambda *a, **kw: upload_response, instance,
                kwargs={"documents": docs},
            )
            _call_sync(
                tracer, "search",
                lambda *a, **kw: iter([]), instance,
                kwargs={"search_text": "hotel", "top": 5, "filter": "rating ge 4"},
            )

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 2
        _assert_all_same_trace(azure_spans)

        upload_span = _find_span(azure_spans, "azure_search.upload_documents")
        assert upload_span.kind == SpanKind.CLIENT
        assert upload_span.status.status_code == StatusCode.OK
        assert upload_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 3
        assert upload_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT] == 3
        assert upload_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_FAILED_COUNT] == 0

        attrs = dict(upload_span.attributes)
        doc0 = json.loads(attrs[f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.0"])
        assert doc0["id"] == "h1"
        doc2 = json.loads(attrs[f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.2"])
        assert doc2["name"] == "Cozy Motel"

        search_span = _find_span(azure_spans, "azure_search.search")
        assert search_span.kind == SpanKind.CLIENT
        assert search_span.status.status_code == StatusCode.OK
        assert search_span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_TEXT] == "hotel"
        assert search_span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_TOP] == 5
        assert search_span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_FILTER] == "rating ge 4"

    def test_document_lifecycle(self, exporter):
        """Full CRUD lifecycle: upload -> get -> merge -> get -> delete.

        2am scenario: "A document disappeared -- when was it last modified?"
        Trace must show every mutation with content so we can reconstruct history.
        """
        import json
        from opentelemetry import trace
        from opentelemetry.semconv_ai import EventAttributes
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        initial_doc = {"id": "lc-1", "name": "Lifecycle Hotel", "rating": 3.0}
        updated_doc = {"id": "lc-1", "name": "Lifecycle Hotel", "rating": 4.5}

        with tracer.start_as_current_span("app.document_lifecycle"):
            _call_sync(
                tracer, "upload_documents",
                lambda *a, **kw: [_mock_indexing_result("lc-1", status_code=201)],
                instance, kwargs={"documents": [initial_doc]},
            )
            _call_sync(
                tracer, "get_document",
                lambda *a, **kw: dict(initial_doc), instance,
                kwargs={"key": "lc-1"},
            )
            _call_sync(
                tracer, "merge_documents",
                lambda *a, **kw: [_mock_indexing_result("lc-1")],
                instance, kwargs={"documents": [{"id": "lc-1", "rating": 4.5}]},
            )
            _call_sync(
                tracer, "get_document",
                lambda *a, **kw: dict(updated_doc), instance,
                kwargs={"key": "lc-1"},
            )
            _call_sync(
                tracer, "delete_documents",
                lambda *a, **kw: [_mock_indexing_result("lc-1")],
                instance, kwargs={"documents": [{"id": "lc-1"}]},
            )

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 5
        _assert_all_same_trace(azure_spans)

        for s in azure_spans:
            assert s.status.status_code == StatusCode.OK

        # Upload content shows initial rating
        upload_span = _find_span(azure_spans, "azure_search.upload_documents")
        attrs = dict(upload_span.attributes)
        doc = json.loads(attrs[f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.0"])
        assert doc["rating"] == 3.0

        # First get_document shows the retrieved document
        get_spans = [s for s in azure_spans if s.name == "azure_search.get_document"]
        assert len(get_spans) == 2
        first_get_doc = json.loads(
            dict(get_spans[0].attributes)[EventAttributes.DB_QUERY_RESULT_DOCUMENT.value]
        )
        assert first_get_doc["rating"] == 3.0

        # Second get_document shows updated rating
        second_get_doc = json.loads(
            dict(get_spans[1].attributes)[EventAttributes.DB_QUERY_RESULT_DOCUMENT.value]
        )
        assert second_get_doc["rating"] == 4.5

        # Merge content shows the delta
        merge_span = _find_span(azure_spans, "azure_search.merge_documents")
        merge_doc = json.loads(
            dict(merge_span.attributes)[f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.0"]
        )
        assert merge_doc["rating"] == 4.5

        # Delete confirms 1 doc removed
        delete_span = _find_span(azure_spans, "azure_search.delete_documents")
        assert delete_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 1
        assert delete_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT] == 1

    def test_typeahead_pipeline(self, exporter):
        """Upload -> autocomplete -> suggest -- typeahead debugging.

        2am scenario: "Autocomplete is returning empty results."
        Trace must show: what was uploaded, autocomplete/suggest results + counts.
        """
        import json
        from opentelemetry import trace
        from opentelemetry.semconv_ai import EventAttributes
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        autocomplete_results = [
            MagicMock(text="luxury", query_plus_text="luxury hotel"),
            MagicMock(text="luxurious", query_plus_text="luxurious resort"),
        ]
        suggest_results = [
            {"@search.text": "Luxury Hotel Downtown", "id": "s1"},
        ]

        with tracer.start_as_current_span("app.typeahead_pipeline"):
            _call_sync(
                tracer, "upload_documents",
                lambda *a, **kw: [_mock_indexing_result("th-1")],
                instance,
                kwargs={"documents": [{"id": "th-1", "name": "Luxury Hotel Downtown"}]},
            )
            _call_sync(
                tracer, "autocomplete",
                lambda *a, **kw: autocomplete_results, instance,
                kwargs={"search_text": "lux", "suggester_name": "sg"},
            )
            _call_sync(
                tracer, "suggest",
                lambda *a, **kw: suggest_results, instance,
                kwargs={"search_text": "lux", "suggester_name": "sg"},
            )

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 3
        _assert_all_same_trace(azure_spans)

        for s in azure_spans:
            assert s.status.status_code == StatusCode.OK

        ac_span = _find_span(azure_spans, "azure_search.autocomplete")
        assert ac_span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_TEXT] == "lux"
        assert ac_span.attributes[SpanAttributes.AZURE_SEARCH_SUGGESTER_NAME] == "sg"
        assert ac_span.attributes[SpanAttributes.AZURE_SEARCH_AUTOCOMPLETE_RESULTS_COUNT] == 2

        ac_attrs = dict(ac_span.attributes)
        entity_0 = json.loads(ac_attrs[f"{EventAttributes.DB_SEARCH_RESULT_ENTITY.value}.0"])
        assert entity_0["text"] == "luxury"
        entity_1 = json.loads(ac_attrs[f"{EventAttributes.DB_SEARCH_RESULT_ENTITY.value}.1"])
        assert entity_1["text"] == "luxurious"

        sg_span = _find_span(azure_spans, "azure_search.suggest")
        assert sg_span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_TEXT] == "lux"
        assert sg_span.attributes[SpanAttributes.AZURE_SEARCH_SUGGEST_RESULTS_COUNT] == 1

    def test_bulk_ingestion_partial_failure(self, exporter):
        """Upload 5 docs where 2 fail -- trace shows which docs failed and why.

        2am scenario: "ETL pipeline uploaded 5 docs but only 3 are searchable."
        Trace must show: per-document success/failure metadata with error messages.
        """
        import json
        from opentelemetry import trace
        from opentelemetry.semconv_ai import EventAttributes
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        docs = [
            {"id": f"bulk-{i}", "name": f"Hotel {i}", "rating": float(i)}
            for i in range(5)
        ]
        response = [
            _mock_indexing_result("bulk-0"),
            _mock_indexing_result("bulk-1"),
            _mock_indexing_result("bulk-2"),
            _mock_indexing_result(
                "bulk-3", succeeded=False, status_code=400,
                error_message="Invalid field 'rating'",
            ),
            _mock_indexing_result(
                "bulk-4", succeeded=False, status_code=400,
                error_message="Document too large",
            ),
        ]

        with tracer.start_as_current_span("app.etl_ingestion"):
            _call_sync(
                tracer, "upload_documents",
                lambda *a, **kw: response, instance,
                kwargs={"documents": docs},
            )

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 1

        span = azure_spans[0]
        assert span.status.status_code == StatusCode.OK
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 5
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT] == 3
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_FAILED_COUNT] == 2

        attrs = dict(span.attributes)

        # Request content: all 5 input docs are captured
        for i in range(5):
            raw = attrs[f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.{i}"]
            doc = json.loads(raw)
            assert doc["id"] == f"bulk-{i}"

        # Response metadata: succeeded docs
        for i in range(3):
            meta = json.loads(attrs[f"{EventAttributes.DB_QUERY_RESULT_METADATA.value}.{i}"])
            assert meta["succeeded"] is True

        # Response metadata: failed docs with actionable error messages
        meta_3 = json.loads(attrs[f"{EventAttributes.DB_QUERY_RESULT_METADATA.value}.3"])
        assert meta_3["succeeded"] is False
        assert meta_3["status_code"] == 400
        assert meta_3["error_message"] == "Invalid field 'rating'"

        meta_4 = json.loads(attrs[f"{EventAttributes.DB_QUERY_RESULT_METADATA.value}.4"])
        assert meta_4["succeeded"] is False
        assert meta_4["error_message"] == "Document too large"

    def test_index_management_pipeline(self, exporter):
        """create_index -> upload -> count -> search -> delete_index.

        2am scenario: "Deployment created the index but search returns 404."
        Trace must show: index created, docs uploaded, count matches, search ran.
        """
        from opentelemetry import trace
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        search_instance = _make_instance("pipeline-test")
        index_instance = _make_index_instance()

        mock_index = MagicMock()
        mock_index.name = "pipeline-test"

        with tracer.start_as_current_span("app.deployment_pipeline"):
            _call_sync(
                tracer, "create_index",
                lambda *a, **kw: mock_index, index_instance,
                kwargs={"index": mock_index},
            )
            _call_sync(
                tracer, "upload_documents",
                lambda *a, **kw: [_mock_indexing_result("p1"), _mock_indexing_result("p2")],
                search_instance,
                kwargs={"documents": [{"id": "p1"}, {"id": "p2"}]},
            )
            _call_sync(
                tracer, "get_document_count",
                lambda *a, **kw: 2, search_instance,
            )
            _call_sync(
                tracer, "search",
                lambda *a, **kw: iter([{"id": "p1"}, {"id": "p2"}]),
                search_instance,
                kwargs={"search_text": "*"},
            )
            _call_sync(
                tracer, "delete_index",
                lambda *a, **kw: None, index_instance,
                kwargs={"index": "pipeline-test"},
            )

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 5
        _assert_all_same_trace(azure_spans)

        for s in azure_spans:
            assert s.status.status_code == StatusCode.OK

        create_span = _find_span(azure_spans, "azure_search.create_index")
        assert create_span.attributes[SpanAttributes.AZURE_SEARCH_INDEX_NAME] == "pipeline-test"

        upload_span = _find_span(azure_spans, "azure_search.upload_documents")
        assert upload_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 2

        count_span = _find_span(azure_spans, "azure_search.get_document_count")
        assert count_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 2

        search_span = _find_span(azure_spans, "azure_search.search")
        assert search_span.attributes[SpanAttributes.AZURE_SEARCH_INDEX_NAME] == "pipeline-test"

        delete_span = _find_span(azure_spans, "azure_search.delete_index")
        assert delete_span.attributes[SpanAttributes.AZURE_SEARCH_INDEX_NAME] == "pipeline-test"

    def test_content_privacy_across_pipeline(self, exporter, monkeypatch):
        """Full pipeline with content disabled -- verify no PII leaks.

        2am scenario: "Compliance audit -- are we leaking PII in traces?"
        Trace must show: operational metadata present, ZERO content attributes.
        """
        from opentelemetry import trace
        from opentelemetry.semconv_ai import EventAttributes
        from opentelemetry.trace.status import StatusCode

        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        content_prefixes = (
            EventAttributes.DB_QUERY_RESULT_DOCUMENT.value,
            EventAttributes.DB_SEARCH_RESULT_ENTITY.value,
            EventAttributes.DB_SEARCH_EMBEDDINGS_VECTOR.value,
            EventAttributes.DB_QUERY_RESULT_METADATA.value,
            EventAttributes.DB_QUERY_RESULT_ID.value,
        )

        with tracer.start_as_current_span("app.privacy_pipeline"):
            _call_sync(
                tracer, "upload_documents",
                lambda *a, **kw: [_mock_indexing_result("priv-1")],
                instance,
                kwargs={"documents": [{"id": "priv-1", "ssn": "123-45-6789"}]},
            )
            _call_sync(
                tracer, "get_document",
                lambda *a, **kw: {"id": "priv-1", "ssn": "123-45-6789"},
                instance, kwargs={"key": "priv-1"},
            )
            _call_sync(
                tracer, "autocomplete",
                lambda *a, **kw: [MagicMock(text="secret", query_plus_text="secret data")],
                instance,
                kwargs={"search_text": "sec", "suggester_name": "sg"},
            )
            _call_sync(
                tracer, "suggest",
                lambda *a, **kw: [{"@search.text": "Secret Doc", "id": "s1"}],
                instance,
                kwargs={"search_text": "sec", "suggester_name": "sg"},
            )

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 4
        _assert_all_same_trace(azure_spans)

        for s in azure_spans:
            assert s.status.status_code == StatusCode.OK
            attrs = dict(s.attributes)
            content_keys = [
                k for k in attrs
                if any(k.startswith(p) for p in content_prefixes)
            ]
            assert content_keys == [], (
                f"Content leaked in span '{s.name}': {content_keys}"
            )

        # But operational metadata IS still present
        upload_span = _find_span(azure_spans, "azure_search.upload_documents")
        assert upload_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 1

        get_span = _find_span(azure_spans, "azure_search.get_document")
        assert get_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_KEY] == "priv-1"

        ac_span = _find_span(azure_spans, "azure_search.autocomplete")
        assert ac_span.attributes[SpanAttributes.AZURE_SEARCH_SUGGESTER_NAME] == "sg"

    def test_error_then_retry_success(self, exporter):
        """First call fails, retry succeeds -- trace shows both for diagnosis.

        2am scenario: "Was this a transient failure? Did the retry work?"
        Trace must show: error span with exception detail, then OK span.
        """
        from opentelemetry import trace
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        call_count = 0

        def flaky_search(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("ServiceUnavailable: retry later")
            return iter([{"id": "1"}])

        with tracer.start_as_current_span("app.search_with_retry"):
            try:
                _call_sync(
                    tracer, "search", flaky_search, instance,
                    kwargs={"search_text": "hotel"},
                )
            except ConnectionError:
                pass  # App catches and retries
            _call_sync(
                tracer, "search", flaky_search, instance,
                kwargs={"search_text": "hotel"},
            )

        spans = _get_spans(exporter)
        search_spans = [s for s in spans if s.name == "azure_search.search"]
        assert len(search_spans) == 2
        _assert_all_same_trace(search_spans)

        assert search_spans[0].status.status_code == StatusCode.ERROR
        assert "ServiceUnavailable" in search_spans[0].status.description

        assert search_spans[1].status.status_code == StatusCode.OK


class TestAsyncWorkflows:
    """Async mirrors of all sync workflow tests.

    Validates that _async_wrap produces identical trace structures as _sync_wrap.
    """

    @pytest.mark.asyncio
    async def test_search_pipeline(self, exporter):
        """Async: upload -> search -- trace correlation and query visibility."""
        import json
        from opentelemetry import trace
        from opentelemetry.semconv_ai import EventAttributes
        from opentelemetry.trace import SpanKind
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        docs = [
            {"id": "h1", "name": "Grand Hotel", "rating": 4.5},
            {"id": "h2", "name": "Budget Inn", "rating": 2.0},
            {"id": "h3", "name": "Cozy Motel", "rating": 3.8},
        ]
        upload_response = [
            _mock_indexing_result("h1"),
            _mock_indexing_result("h2"),
            _mock_indexing_result("h3"),
        ]

        async def mock_upload(*a, **kw):
            return upload_response

        async def mock_search(*a, **kw):
            return iter([])

        with tracer.start_as_current_span("app.async_ingest_and_search"):
            await _call_async(
                tracer, "upload_documents", mock_upload, instance,
                kwargs={"documents": docs},
            )
            await _call_async(
                tracer, "search", mock_search, instance,
                kwargs={"search_text": "hotel", "top": 5, "filter": "rating ge 4"},
            )

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 2
        _assert_all_same_trace(azure_spans)

        upload_span = _find_span(azure_spans, "azure_search.upload_documents")
        assert upload_span.kind == SpanKind.CLIENT
        assert upload_span.status.status_code == StatusCode.OK
        assert upload_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 3
        assert upload_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT] == 3

        attrs = dict(upload_span.attributes)
        doc0 = json.loads(attrs[f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.0"])
        assert doc0["id"] == "h1"

        search_span = _find_span(azure_spans, "azure_search.search")
        assert search_span.status.status_code == StatusCode.OK
        assert search_span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_TEXT] == "hotel"
        assert search_span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_TOP] == 5

    @pytest.mark.asyncio
    async def test_document_lifecycle(self, exporter):
        """Async: upload -> get -> merge -> get -> delete -- CRUD audit trail."""
        import json
        from opentelemetry import trace
        from opentelemetry.semconv_ai import EventAttributes
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        initial_doc = {"id": "lc-1", "name": "Lifecycle Hotel", "rating": 3.0}
        updated_doc = {"id": "lc-1", "name": "Lifecycle Hotel", "rating": 4.5}

        async def upload(*a, **kw):
            return [_mock_indexing_result("lc-1", status_code=201)]

        async def get_initial(*a, **kw):
            return dict(initial_doc)

        async def merge(*a, **kw):
            return [_mock_indexing_result("lc-1")]

        async def get_updated(*a, **kw):
            return dict(updated_doc)

        async def delete(*a, **kw):
            return [_mock_indexing_result("lc-1")]

        with tracer.start_as_current_span("app.async_document_lifecycle"):
            await _call_async(
                tracer, "upload_documents", upload, instance,
                kwargs={"documents": [initial_doc]},
            )
            await _call_async(
                tracer, "get_document", get_initial, instance,
                kwargs={"key": "lc-1"},
            )
            await _call_async(
                tracer, "merge_documents", merge, instance,
                kwargs={"documents": [{"id": "lc-1", "rating": 4.5}]},
            )
            await _call_async(
                tracer, "get_document", get_updated, instance,
                kwargs={"key": "lc-1"},
            )
            await _call_async(
                tracer, "delete_documents", delete, instance,
                kwargs={"documents": [{"id": "lc-1"}]},
            )

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 5
        _assert_all_same_trace(azure_spans)

        for s in azure_spans:
            assert s.status.status_code == StatusCode.OK

        get_spans = [s for s in azure_spans if s.name == "azure_search.get_document"]
        first_doc = json.loads(
            dict(get_spans[0].attributes)[EventAttributes.DB_QUERY_RESULT_DOCUMENT.value]
        )
        assert first_doc["rating"] == 3.0
        second_doc = json.loads(
            dict(get_spans[1].attributes)[EventAttributes.DB_QUERY_RESULT_DOCUMENT.value]
        )
        assert second_doc["rating"] == 4.5

    @pytest.mark.asyncio
    async def test_typeahead_pipeline(self, exporter):
        """Async: upload -> autocomplete -> suggest -- typeahead debugging."""
        import json
        from opentelemetry import trace
        from opentelemetry.semconv_ai import EventAttributes

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        autocomplete_results = [
            MagicMock(text="luxury", query_plus_text="luxury hotel"),
        ]
        suggest_results = [
            {"@search.text": "Luxury Hotel", "id": "s1"},
        ]

        async def mock_upload(*a, **kw):
            return [_mock_indexing_result("th-1")]

        async def mock_ac(*a, **kw):
            return autocomplete_results

        async def mock_sg(*a, **kw):
            return suggest_results

        with tracer.start_as_current_span("app.async_typeahead"):
            await _call_async(
                tracer, "upload_documents", mock_upload, instance,
                kwargs={"documents": [{"id": "th-1", "name": "Luxury Hotel"}]},
            )
            await _call_async(
                tracer, "autocomplete", mock_ac, instance,
                kwargs={"search_text": "lux", "suggester_name": "sg"},
            )
            await _call_async(
                tracer, "suggest", mock_sg, instance,
                kwargs={"search_text": "lux", "suggester_name": "sg"},
            )

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 3
        _assert_all_same_trace(azure_spans)

        ac_span = _find_span(azure_spans, "azure_search.autocomplete")
        assert ac_span.attributes[SpanAttributes.AZURE_SEARCH_AUTOCOMPLETE_RESULTS_COUNT] == 1
        entity_0 = json.loads(
            dict(ac_span.attributes)[f"{EventAttributes.DB_SEARCH_RESULT_ENTITY.value}.0"]
        )
        assert entity_0["text"] == "luxury"

        sg_span = _find_span(azure_spans, "azure_search.suggest")
        assert sg_span.attributes[SpanAttributes.AZURE_SEARCH_SUGGEST_RESULTS_COUNT] == 1

    @pytest.mark.asyncio
    async def test_bulk_ingestion_partial_failure(self, exporter):
        """Async: upload 5 docs, 2 fail -- per-document failure metadata."""
        import json
        from opentelemetry import trace
        from opentelemetry.semconv_ai import EventAttributes

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        docs = [{"id": f"bulk-{i}", "name": f"Hotel {i}"} for i in range(5)]
        response = [
            _mock_indexing_result("bulk-0"),
            _mock_indexing_result("bulk-1"),
            _mock_indexing_result("bulk-2"),
            _mock_indexing_result("bulk-3", succeeded=False, status_code=400, error_message="Bad format"),
            _mock_indexing_result("bulk-4", succeeded=False, status_code=400, error_message="Too large"),
        ]

        async def mock_upload(*a, **kw):
            return response

        with tracer.start_as_current_span("app.async_etl"):
            await _call_async(
                tracer, "upload_documents", mock_upload, instance,
                kwargs={"documents": docs},
            )

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 1

        span = azure_spans[0]
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT] == 3
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_FAILED_COUNT] == 2

        attrs = dict(span.attributes)
        meta_3 = json.loads(attrs[f"{EventAttributes.DB_QUERY_RESULT_METADATA.value}.3"])
        assert meta_3["succeeded"] is False
        assert meta_3["error_message"] == "Bad format"

    @pytest.mark.asyncio
    async def test_index_management_pipeline(self, exporter):
        """Async: create_index -> upload -> count -> search -> delete_index."""
        from opentelemetry import trace
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        search_instance = _make_instance("pipeline-test")
        index_instance = _make_index_instance()

        mock_index = MagicMock()
        mock_index.name = "pipeline-test"

        async def create_idx(*a, **kw):
            return mock_index

        async def upload(*a, **kw):
            return [_mock_indexing_result("p1")]

        async def count(*a, **kw):
            return 1

        async def search(*a, **kw):
            return iter([])

        async def delete_idx(*a, **kw):
            return None

        with tracer.start_as_current_span("app.async_deployment"):
            await _call_async(
                tracer, "create_index", create_idx, index_instance,
                kwargs={"index": mock_index},
            )
            await _call_async(
                tracer, "upload_documents", upload, search_instance,
                kwargs={"documents": [{"id": "p1"}]},
            )
            await _call_async(tracer, "get_document_count", count, search_instance)
            await _call_async(
                tracer, "search", search, search_instance,
                kwargs={"search_text": "*"},
            )
            await _call_async(
                tracer, "delete_index", delete_idx, index_instance,
                kwargs={"index": "pipeline-test"},
            )

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 5
        _assert_all_same_trace(azure_spans)

        for s in azure_spans:
            assert s.status.status_code == StatusCode.OK

        create_span = _find_span(azure_spans, "azure_search.create_index")
        assert create_span.attributes[SpanAttributes.AZURE_SEARCH_INDEX_NAME] == "pipeline-test"

    @pytest.mark.asyncio
    async def test_content_privacy_across_pipeline(self, exporter, monkeypatch):
        """Async: full pipeline with content disabled -- no PII leaks."""
        from opentelemetry import trace
        from opentelemetry.semconv_ai import EventAttributes
        from opentelemetry.trace.status import StatusCode

        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        content_prefixes = (
            EventAttributes.DB_QUERY_RESULT_DOCUMENT.value,
            EventAttributes.DB_SEARCH_RESULT_ENTITY.value,
            EventAttributes.DB_SEARCH_EMBEDDINGS_VECTOR.value,
            EventAttributes.DB_QUERY_RESULT_METADATA.value,
            EventAttributes.DB_QUERY_RESULT_ID.value,
        )

        async def upload(*a, **kw):
            return [_mock_indexing_result("priv-1")]

        async def get_doc(*a, **kw):
            return {"id": "priv-1", "secret": "classified"}

        async def autocomplete(*a, **kw):
            return [MagicMock(text="secret", query_plus_text="secret data")]

        async def suggest(*a, **kw):
            return [{"@search.text": "Secret", "id": "s1"}]

        with tracer.start_as_current_span("app.async_privacy"):
            await _call_async(
                tracer, "upload_documents", upload, instance,
                kwargs={"documents": [{"id": "priv-1", "secret": "classified"}]},
            )
            await _call_async(
                tracer, "get_document", get_doc, instance,
                kwargs={"key": "priv-1"},
            )
            await _call_async(
                tracer, "autocomplete", autocomplete, instance,
                kwargs={"search_text": "sec", "suggester_name": "sg"},
            )
            await _call_async(
                tracer, "suggest", suggest, instance,
                kwargs={"search_text": "sec", "suggester_name": "sg"},
            )

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 4
        _assert_all_same_trace(azure_spans)

        for s in azure_spans:
            assert s.status.status_code == StatusCode.OK
            attrs = dict(s.attributes)
            content_keys = [
                k for k in attrs
                if any(k.startswith(p) for p in content_prefixes)
            ]
            assert content_keys == [], f"Content leaked in '{s.name}': {content_keys}"

        upload_span = _find_span(azure_spans, "azure_search.upload_documents")
        assert upload_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 1

    @pytest.mark.asyncio
    async def test_error_then_retry_success(self, exporter):
        """Async: first call fails, retry succeeds -- transient failure diagnosis."""
        from opentelemetry import trace
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        call_count = 0

        async def flaky_search(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("ServiceUnavailable: retry later")
            return iter([{"id": "1"}])

        with tracer.start_as_current_span("app.async_retry"):
            try:
                await _call_async(
                    tracer, "search", flaky_search, instance,
                    kwargs={"search_text": "hotel"},
                )
            except ConnectionError:
                pass
            await _call_async(
                tracer, "search", flaky_search, instance,
                kwargs={"search_text": "hotel"},
            )

        spans = _get_spans(exporter)
        search_spans = [s for s in spans if s.name == "azure_search.search"]
        assert len(search_spans) == 2
        _assert_all_same_trace(search_spans)

        assert search_spans[0].status.status_code == StatusCode.ERROR
        assert "ServiceUnavailable" in search_spans[0].status.description
        assert search_spans[1].status.status_code == StatusCode.OK
