# traceloop-sdk

Traceloop's Python SDK allows you to easily start monitoring and debugging your LLM execution. Tracing is done in a non-intrusive way, built on top of OpenTelemetry. You can choose to export the traces to Traceloop, or to your existing observability stack.

## Installation

### Basic Installation (includes OpenAI + Anthropic instrumentation)

```bash
pip install traceloop-sdk
```

### Install with Specific Providers

```bash
# Single provider
pip install 'traceloop-sdk[langchain]'

# Multiple providers
pip install 'traceloop-sdk[openai,langchain,pinecone]'
```

### Full Installation (All Instrumentations)

```bash
pip install 'traceloop-sdk[all]'
```

### Available Extras

**LLM Providers:**
`openai`, `anthropic`, `mistralai`, `cohere`, `google-generativeai`, `bedrock`, `sagemaker`, `vertexai`, `watsonx`, `ollama`, `together`, `groq`, `replicate`, `writer`, `alephalpha`

**Frameworks:**
`langchain`, `llamaindex`, `crewai`, `haystack`, `agno`, `openai-agents`, `mcp`, `transformers`

**Vector Databases and Search:**
`azure-search`, `pinecone`, `qdrant`, `lancedb`, `chromadb`, `milvus`, `marqo`, `weaviate`

## Quick Start

```python
Traceloop.init(app_name="joke_generation_service")

@workflow(name="joke_creation")
def create_joke():
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    return completion.choices[0].message.content
```
