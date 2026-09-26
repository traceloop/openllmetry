# OpenTelemetry Azure AI Search Instrumentation

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This library allows tracing client-side calls to Azure AI Search using OpenTelemetry.

## Installation

```bash
pip install opentelemetry-instrumentation-azure-search
```

## Supported Operations

### SearchClient

| Method | Span Name | Attributes Captured |
|--------|-----------|-------------------|
| `search()` | `azure_search.search` | search_text, top, filter, index_name |
| `get_document()` | `azure_search.get_document` | index_name |
| `autocomplete()` | `azure_search.autocomplete` | autocomplete_text, index_name |
| `suggest()` | `azure_search.suggest` | suggest_text, index_name |
| `index_documents()` | `azure_search.index_documents` | documents_count, succeeded_count |
| `upload_documents()` | `azure_search.upload_documents` | documents_count, succeeded_count |
| `merge_documents()` | `azure_search.merge_documents` | documents_count, succeeded_count |
| `merge_or_upload_documents()` | `azure_search.merge_or_upload_documents` | documents_count, succeeded_count |
| `delete_documents()` | `azure_search.delete_documents` | documents_count, succeeded_count |
| `get_document_count()` | `azure_search.get_document_count` | documents_count |

### SearchIndexClient

| Method | Span Name | Attributes Captured |
|--------|-----------|-------------------|
| `create_index()` | `azure_search.create_index` | index_name |
| `create_or_update_index()` | `azure_search.create_or_update_index` | index_name |
| `delete_index()` | `azure_search.delete_index` | index_name |
| `get_index()` | `azure_search.get_index` | index_name |
| `list_indexes()` | `azure_search.list_indexes` | — |
| `get_index_statistics()` | `azure_search.get_index_statistics` | index_doc_count, index_size_bytes |
| `analyze_text()` | `azure_search.analyze_text` | — |
| `get_service_statistics()` | `azure_search.get_service_statistics` | service_usage, service_limit |

### SearchIndexerClient

| Method | Span Name | Attributes Captured |
|--------|-----------|-------------------|
| `create_indexer()` | `azure_search.create_indexer` | indexer_name |
| `create_or_update_indexer()` | `azure_search.create_or_update_indexer` | indexer_name |
| `delete_indexer()` | `azure_search.delete_indexer` | indexer_name |
| `get_indexer()` | `azure_search.get_indexer` | indexer_name |
| `get_indexers()` | `azure_search.get_indexers` | — |
| `get_indexer_status()` | `azure_search.get_indexer_status` | indexer_status |
| `run_indexer()` | `azure_search.run_indexer` | indexer_name |
| `reset_indexer()` | `azure_search.reset_indexer` | indexer_name |
| `create_data_source_connection()` | `azure_search.create_data_source_connection` | — |
| `create_skillset()` | `azure_search.create_skillset` | skillset_name |
| `get_skillset()` | `azure_search.get_skillset` | skillset_name |
| `delete_skillset()` | `azure_search.delete_skillset` | skillset_name |

## Usage

### Auto-instrumentation via Traceloop SDK

```python
from traceloop.sdk import Traceloop

Traceloop.init(app_name="my-rag-app")
```

### Manual instrumentation

```python
from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor

AzureSearchInstrumentor().instrument()
```

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
