# OpenTelemetry Voyage AI Instrumentation

This library allows tracing Voyage AI API calls with OpenTelemetry.

## Installation

```bash
pip install opentelemetry-instrumentation-voyageai
```

## Usage

```python
from opentelemetry.instrumentation.voyageai import VoyageAIInstrumentor

VoyageAIInstrumentor().instrument()

# Now use Voyage AI as usual
import voyageai

client = voyageai.Client()

# Embeddings
result = client.embed(texts=["Hello, world!"], model="voyage-3")

# Reranking
result = client.rerank(
    query="What is the capital of France?",
    documents=["Paris is the capital of France.", "London is in England."],
    model="rerank-2.5"
)
```

## Semantic Conventions

This instrumentation follows the OpenTelemetry GenAI semantic conventions:

- `gen_ai.system`: "voyageai"
- `gen_ai.operation.name`: "embeddings" or "rerank"
- `gen_ai.request.model`: The model name
- `gen_ai.usage.input_tokens`: Token count from the response
- `gen_ai.embeddings.dimension.count`: Embedding vector dimension (for embed only)
