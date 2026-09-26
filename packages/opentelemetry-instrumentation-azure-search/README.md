# OpenTelemetry Azure AI Search Instrumentation

This package provides OpenTelemetry instrumentation for [Azure AI Search](https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search) (formerly Azure Cognitive Search).

## Installation

```bash
pip install opentelemetry-instrumentation-azure-search
```

Or with the Azure Search SDK included:

```bash
pip install 'opentelemetry-instrumentation-azure-search[instruments]'
```

## Usage

### Auto-instrumentation via Traceloop SDK

```python
from traceloop.sdk import Traceloop

Traceloop.init(app_name="my_app")
```

### Manual instrumentation

```python
from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor

AzureSearchInstrumentor().instrument()
```

## Instrumented methods

| Method | Span name |
|---|---|
| `SearchClient.search()` | `azure_search.search` |
| `SearchClient.upload_documents()` | `azure_search.upload_documents` |
| `SearchClient.merge_documents()` | `azure_search.merge_documents` |
| `SearchClient.merge_or_upload_documents()` | `azure_search.merge_or_upload_documents` |
| `SearchClient.delete_documents()` | `azure_search.delete_documents` |

## Span attributes

| Attribute | Description |
|---|---|
| `db.system` | Always `azure_search` |
| `server.address` | Azure Search endpoint URL |
| `azure_search.index_name` | Name of the index being queried |
| `azure_search.search_text` | The search query string |
| `azure_search.top` | Max results requested |
| `azure_search.filter` | OData filter expression |
| `azure_search.duration` | Operation duration in seconds |
| `azure_search.affected_documents` | Number of documents in write operations |
| `azure_search.succeeded_documents` | Number of successfully written documents |