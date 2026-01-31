# OpenTelemetry Azure AI Search Instrumentation

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

Or install with the azure-search extra:

```bash
pip install 'traceloop-sdk[azure-search]'
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
- `get_index_statistics()` - Get index statistics
- `analyze_text()` - Analyze text with an analyzer

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
| `azure_search.document.count` | Number of documents in operation |
| `azure_search.suggester_name` | Suggester name for autocomplete/suggest |

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
