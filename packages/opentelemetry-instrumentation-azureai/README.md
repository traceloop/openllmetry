# OpenTelemetry Azure AI Search Instrumentation

This package provides OpenTelemetry instrumentation for the Azure AI Search Python client library. It automatically captures telemetry data from Azure AI Search operations, including search queries, durations, result counts, and errors.

## Installation

```bash
pip install opentelemetry-instrumentation-azure-search
```
C:\Users\Mr.Imperium\Documents\GitHub\openllmetry\packages\opentelemetry-instrumentation-azureai
## Requirements

- Python 3.7 or later
- `azure-search-documents >= 11.4.0`
- `opentelemetry-api`
- `opentelemetry-instrumentation`

## Features

- Automatic instrumentation of Azure AI Search client operations
- Spans for search, suggest, and autocomplete operations
- Metrics for operation duration, result counts, and exceptions
- Detailed operation attributes including query parameters
- Integration with OpenTelemetry semantic conventions for AI/ML

## Usage

### Basic Setup

```python
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor

# Initialize the instrumentor
AzureSearchInstrumentor().instrument()

# Your existing Azure Search client code continues to work as normal
search_client = SearchClient(
    endpoint="<search-endpoint>",
    index_name="<index-name>",
    credential=AzureKeyCredential("<api-key>")
)

# Operations will now automatically generate telemetry
results = search_client.search(search_text="example query")
```

### Custom Configuration

```python
from opentelemetry import trace
from opentelemetry import metrics
from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor

# Configure custom trace and metric providers
tracer_provider = trace.TracerProvider()
meter_provider = metrics.MeterProvider()

# Initialize with custom providers
AzureSearchInstrumentor().instrument(
    tracer_provider=tracer_provider,
    meter_provider=meter_provider
)
```

## Telemetry Details

### Spans

Each search operation creates a span with the following attributes:

- `llm.system`: "Azure AI Search"
- `llm.request.type`: "search"
- `azure.search.index_name`: Name of the search index
- `azure.search.query`: Search query text (when available)
- `azure.search.filter`: Applied filters (when available)
- `azure.search.facets`: Requested facets (when available)
- `azure.search.top`: Number of results requested (when available)
- `azure.search.result_count`: Number of results returned (when available)

### Metrics

The following metrics are collected:

| Name | Type | Description | Unit |
|------|------|-------------|------|
| `llm.operation.duration` | Histogram | Duration of search operations | seconds |
| `llm.azure_search.results` | Counter | Number of results returned | result |
| `llm.azure_search.exceptions` | Counter | Number of exceptions occurred | exception |

