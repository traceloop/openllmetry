# OpenLLMetry Azure AI Search Integration

## Overview

This package provides an integration for Azure AI Search within the OpenLLMetry observability framework. It enables detailed tracing and metrics collection for search operations.

## Installation

```bash
pip install openllmetry-azure-search
```

Or with Poetry:

```bash
poetry add openllmetry-azure-search
```

## Usage

```python
from openllmetry.integrations.azure_search import AzureSearchIntegration

# Initialize the integration
search = AzureSearchIntegration(
    endpoint="https://your-search-service.search.windows.net",
    key="your-api-key",
    index_name="your-index-name"
)

# Perform a search
results = search.search(
    query="your search query",
    filter="category eq 'books'",
    top=10
)
```

## Features

- Automatic OpenTelemetry span creation for search operations
- Comprehensive metrics collection
- Error tracking and status reporting
- Flexible configuration options

## Requirements

- Python 3.8+
- Azure AI Search
- OpenTelemetry
