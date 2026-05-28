# sample-app

Runnable examples that exercise OpenLLMetry against real providers — useful as smoke tests and as starting points for your own code.

## Start here if you're new

[`sample_app/beginner_tracing_example.py`](sample_app/beginner_tracing_example.py) is a short walkthrough that covers `Traceloop.init`, `@workflow`, and `@task` against a single OpenAI call.

```bash
export OPENAI_API_KEY=sk-...
poetry run python sample_app/beginner_tracing_example.py
```

Spans print to stdout by default. Point the standard `OTEL_EXPORTER_OTLP_*` env vars at a backend (Jaeger, Honeycomb, Datadog, Traceloop, …) to ship them.

## Everything else

The other files follow the same pattern but for specific providers and scenarios (Anthropic, Bedrock, Cohere, Pinecone, Chroma, agents, streaming, structured outputs, …). Pick whichever is closest to what you're building.
