# OpenTelemetry Pinecone Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-pinecone/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-pinecone.svg">
</a>

This library allows tracing client-side calls to Pinecone vector DB sent with the official [Pinecone library](https://github.com/pinecone-io/pinecone-python-client).

## Installation

### Prerequisites

You need to install the official Pinecone Python SDK:

```bash
pip install pinecone
```

**Note:** This instrumentation supports the new `pinecone` package (v2.2.3+). If you're upgrading from the legacy `pinecone-client` package, please:

1. Uninstall the old package: `pip uninstall pinecone-client`
2. Install the new package: `pip install pinecone`

### Install the Instrumentation

```bash
pip install opentelemetry-instrumentation-pinecone
```

## Example usage

### Basic Setup

```python
from opentelemetry.instrumentation.pinecone import PineconeInstrumentor

# Instrument Pinecone
PineconeInstrumentor().instrument()
```

### Complete Example

```python
import os
from pinecone import Pinecone
from opentelemetry.instrumentation.pinecone import PineconeInstrumentor

# Enable instrumentation
PineconeInstrumentor().instrument()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Use Pinecone normally - all operations will be traced
index = pc.Index("your-index-name")
result = index.query(
    vector=[0.1, 0.2, 0.3],
    top_k=10,
    include_metadata=True
)
```

## Supported Operations

This instrumentation automatically traces:
- `index.query()` - Vector similarity queries
- `index.upsert()` - Vector insertions/updates  
- `index.delete()` - Vector deletions

## Migration from pinecone-client

If you're migrating from the legacy `pinecone-client` package:

```bash
# Remove old package
pip uninstall pinecone-client pinecone-plugin-inference

# Install new package
pip install pinecone

# Update your imports (if needed)
# Old: from pinecone import Client
# New: from pinecone import Pinecone
```

The API remains largely compatible, but refer to the [official Pinecone migration guide](https://github.com/pinecone-io/pinecone-python-client) for detailed changes.
