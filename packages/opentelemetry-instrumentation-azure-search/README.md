# OpenTelemetry Azure AI Search Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-azure-search/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-azure-search.svg" alt="PyPI version">
</a>

This library provides automatic instrumentation for the [Azure AI Search](https://learn.microsoft.com/en-us/azure/search/) Python SDK (`azure-search-documents`).

## Installation

```bash
pip install opentelemetry-instrumentation-azure-search
```

## Usage

### Automatic Instrumentation

```python
from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor

AzureSearchInstrumentor().instrument()
```

### With Traceloop SDK

```python
from traceloop.sdk import Traceloop

Traceloop.init(app_name="my-app")
# Azure Search instrumentation is automatically enabled
```

## Instrumented Operations

### SearchClient

- `search()` - Full-text and vector search
- `get_document()` - Retrieve a document by key
- `get_document_count()` - Get document count in index
- `upload_documents()` - Upload documents to index
- `merge_documents()` - Merge documents in index
- `delete_documents()` - Delete documents from index
- `merge_or_upload_documents()` - Merge or upload documents
- `index_documents()` - Batch indexing operations
- `autocomplete()` - Autocomplete suggestions
- `suggest()` - Search suggestions

### SearchIndexClient

- `create_index()` - Create a new search index
- `create_or_update_index()` - Create or update an index
- `delete_index()` - Delete an index
- `get_index()` - Get index definition
- `list_indexes()` - List all indexes
- `list_index_names()` - List index names only
- `get_index_statistics()` - Get index statistics
- `get_service_statistics()` - Get service-level statistics
- `analyze_text()` - Analyze text with an analyzer
- `create_synonym_map()` - Create a synonym map
- `create_or_update_synonym_map()` - Create or update a synonym map
- `delete_synonym_map()` - Delete a synonym map
- `get_synonym_map()` - Get a synonym map
- `get_synonym_maps()` - List all synonym maps
- `get_synonym_map_names()` - List synonym map names only

### SearchIndexerClient

- `create_indexer()` / `create_or_update_indexer()` / `delete_indexer()` / `get_indexer()` / `get_indexers()` / `get_indexer_names()` - Indexer management
- `run_indexer()` / `reset_indexer()` / `get_indexer_status()` - Indexer operations
- `create_data_source_connection()` / `create_or_update_data_source_connection()` / `delete_data_source_connection()` / `get_data_source_connection()` / `get_data_source_connections()` / `get_data_source_connection_names()` - Data source management
- `create_skillset()` / `create_or_update_skillset()` / `delete_skillset()` / `get_skillset()` / `get_skillsets()` / `get_skillset_names()` - Skillset management

### SearchIndexingBufferedSender

- `upload_documents()` - Buffer documents for upload
- `delete_documents()` - Buffer documents for deletion
- `merge_documents()` - Buffer documents for merge
- `merge_or_upload_documents()` - Buffer documents for merge or upload
- `index_documents()` - Batch indexing operations
- `flush()` - Flush buffered operations

## Span Attributes

| Attribute | Description |
|-----------|-------------|
| `db.system` | Always "Azure AI Search" |
| `azure_search.index_name` | Name of the search index |
| `azure_search.search.text` | Search query text |
| `azure_search.search.top` | Number of results to return |
| `azure_search.search.skip` | Number of results to skip |
| `azure_search.search.filter` | Filter expression |
| `azure_search.search.query_type` | Query type (simple, full, semantic) |
| `azure_search.search.facets` | Facet expressions (comma-joined) |
| `azure_search.search.order_by` | Order-by expressions (comma-joined) |
| `azure_search.search.vector_query_kind` | Vector query type (`vector` or `text`) |
| `azure_search.search.vector_weight` | Vector query weight |
| `azure_search.search.vector_oversampling` | Vector query oversampling factor |
| `azure_search.document.count` | Number of documents in operation |
| `azure_search.suggester_name` | Suggester name for autocomplete/suggest |
| `azure_search.synonym_map.name` | Name of the synonym map |
| `azure_search.synonym_map.synonyms_count` | Number of synonym rules |
| `azure_search.service.document_count` | Service-level document count |
| `azure_search.service.index_count` | Service-level index count |

## Content Capture

By default, the instrumentation captures response and request content (documents, autocomplete suggestions, vector embeddings) as **indexed span attributes** (e.g., `db.query.result.document.0`, `db.search.result.entity.0`). This follows the same pattern as the LLM instrumentations (`gen_ai_prompt.0.content`) and ensures content is visible in APM backends like Elastic APM.

This can be controlled via the `TRACELOOP_TRACE_CONTENT` environment variable.

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TRACELOOP_TRACE_CONTENT` | `true` | Enable/disable content capture in span attributes |

Accepted truthy values: `true`, `1`, `yes`, `on` (case-insensitive).

### Per-Request Override

You can override the environment variable on a per-request basis using the OpenTelemetry context API:

```python
from opentelemetry import context as context_api

# Enable content tracing for a specific request even if globally disabled
ctx = context_api.set_value("override_enable_content_tracing", True)
token = context_api.attach(ctx)
try:
    results = client.get_document(key="doc-1")
finally:
    context_api.detach(token)
```

### What Content is Captured

| Operation | Attribute Pattern | Content |
|-----------|------------------|---------|
| `get_document()` | `db.query.result.document` | Full document JSON |
| `autocomplete()` | `db.search.result.entity.{index}` | Each suggestion (text + query_plus_text) |
| `suggest()` | `db.search.result.entity.{index}` | Each suggestion item |
| `upload/merge/delete_documents()` | `db.query.result.document.{index}` (request), `db.query.result.metadata.{index}` (response) | Each input document + indexing result metadata |
| `index_documents()` | `db.query.result.document.{index}` (request), `db.query.result.metadata.{index}` (response) | Each batch action + indexing result metadata |
| `search()` with `vector_queries` | `db.search.embeddings.vector.{index}` | Vector or text from each vector query |

**Note:** `search()` results are not captured because `SearchItemPaged` is a lazy iterator — consuming it would break user code. The `get_document_count()` response is an integer already captured as a span attribute.

## Example

```python
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor

# Initialize instrumentation
AzureSearchInstrumentor().instrument()

# Create client
client = SearchClient(
    endpoint="https://my-search.search.windows.net",
    index_name="hotels",
    credential=AzureKeyCredential("api-key")
)

# Search operations are automatically traced
results = client.search(
    search_text="luxury hotel",
    filter="rating ge 4",
    top=10
)
```

## Developer Guide

For developers looking to extend this instrumentation or understand how span attributes are extracted:

📖 **[Span Attribute Extraction Guide](docs/SPAN_ATTRIBUTES_GUIDE.md)**

This comprehensive guide covers:
- How span attributes are extracted from SDK method calls
- Step-by-step methodology for adding new SDK methods
- Best practices and common patterns
- Testing strategies with VCR cassettes
- Troubleshooting tips
