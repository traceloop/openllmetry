# Beginner LLM Tracing Example

A step-by-step example showing how to trace an LLM call using OpenLLMetry (Traceloop SDK). No API key or external service needed — all traces print to your console.

## What you'll learn

1. **What is LLM tracing?** Recording metadata about each LLM call (prompt, response, tokens, latency) for debugging and monitoring.
2. **Why observability matters** — understand what your LLM app is doing in production.
3. **How OpenLLMetry helps** — automatic tracing with decorators or manual tracing with context managers.
4. **How spans work** — each LLM call becomes a span; spans link into traces via `trace_id` / `span_id`.

## Prerequisites

- Python 3.10+
- The `sample-app` package installed in this repo (run `npx nx run sample-app:install` from the repo root)

## How to run

From the `packages/sample-app` directory:

```bash
uv run python -m sample_app.alephalpha_example.traced_llm_call
```

Or run the file directly:

```bash
uv run python sample_app/alephalpha_example/traced_llm_call.py
```

## What to expect

The script prints three sections:

1. **Manual tracing** — Uses `track_llm_call()` to manually record request, response, and usage for a single LLM call.
2. **Decorator-based tracing** — Uses `@task` and `@workflow` decorators to automatically create spans.
3. **Explanation** — A summary of what you should see in the console output.

### Console output

You'll see JSON blobs printed before each print statement. Each JSON object is an OpenTelemetry **span**. Note: the `beginner_tracing_example` workflow span appears *after* the explanation text printed inside that function (the workflow span closes when the function returns). Look for:

| Field | What it shows |
|-------|--------------|
| `name` | The span name (`openai.chat`, `generate_response`, `beginner_tracing_example`) |
| `attributes.gen_ai.request.model` | Which model was called (`gpt-4o`) |
| `attributes.gen_ai.prompt.0.content` | The user's prompt |
| `attributes.gen_ai.completion.0.content` | The model's response |
| `attributes.gen_ai.usage.input_tokens` / `output_tokens` / `total_tokens` | Token counts |
| `trace_id` / `span_id` | Links spans into a tree |

The `@workflow` span is the parent; `@task` and `track_llm_call` spans are children. This tree structure is what observability tools render as a timeline.

## Key concepts

### Manual tracing (`track_llm_call`)

Use this when you want full control over what's recorded:

```python
with track_llm_call(vendor="openai", type="chat") as span:
    span.report_request(model="gpt-4o", messages=[LLMMessage(role="user", content="Hi")])
    # ... make your LLM call ...
    span.report_response(model="gpt-4o", completions=[response])
    span.report_usage(
        LLMUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
    )
```

### Decorator tracing (`@task`, `@workflow`)

Use this for automatic instrumentation of functions:

```python
@workflow(name="my_app")
def my_app():
    result = call_llm()

@task(name="call_llm")
def call_llm():
    # ... your LLM call here ...
```

### Production setup

Replace `ConsoleSpanExporter()` with the Traceloop platform exporter (no code change needed — just set your API key):

```python
Traceloop.init(
    app_name="my-production-app",
    api_key=os.environ["TRACELOOP_API_KEY"],
)
```

Or point to any OTLP-compatible backend (Jaeger, Grafana, Datadog, Honeycomb, etc.) using the `exporter` parameter.

## Next steps

- Browse the other examples in `sample_app/` to see tracing for different providers (Anthropic, Groq, OpenAI, etc.)
- Read the [OpenLLMetry docs](https://traceloop.com/docs/openllmetry)
- Join the [Traceloop Discord](https://traceloop.com/discord) for help
