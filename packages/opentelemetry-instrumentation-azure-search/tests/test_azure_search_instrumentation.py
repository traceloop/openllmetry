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

    def test_document_batch_from_generator(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_document_batch_attributes
        from opentelemetry import trace

        def doc_generator():
            yield {"id": "1"}
            yield {"id": "2"}
            yield {"id": "3"}

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_document_batch_attributes(span, (), {"documents": doc_generator()})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 3

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


