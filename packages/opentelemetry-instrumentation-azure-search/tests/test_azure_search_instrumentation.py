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
