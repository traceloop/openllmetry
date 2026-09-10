# Sample App

Runnable examples for tracing LLM applications with [OpenLLMetry](https://github.com/traceloop/openllmetry).

This guide walks you through the **Groq** example — a good starting point because Groq offers a free API tier and responses are fast.

## What you'll learn

Running `groq_example.py` shows how OpenLLMetry:

1. Initializes tracing with `Traceloop.init()`
2. Groups your code into a **workflow** and **task** using decorators
3. Automatically records the Groq LLM call as an OpenTelemetry span

## Prerequisites

- Python 3.10–3.12 (see `.python-version`)
- [Node.js](https://nodejs.org/) (for monorepo commands)
- [uv](https://docs.astral.sh/uv/) (Python package manager used by this repo)
- A free [Groq API key](https://console.groq.com/keys)

## Setup

### 1. Install dependencies

From the **repository root**:

```bash
npm ci
npx nx run sample-app:install
```

### 2. Configure your API key

```bash
cd packages/sample-app
cp .env.example .env
```

Edit `.env` and set your Groq key:

```bash
GROQ_API_KEY=gsk-your-key-here
```

Load the environment variables before running the example:

```bash
export $(grep -v '^#' .env | xargs)
```

## Run the Groq example

```bash
cd packages/sample-app
uv run python sample_app/groq_example.py
```

### Expected output

You should see:

1. **Trace JSON** printed to your terminal (spans for the workflow, task, and Groq chat call)
2. **A joke** about OpenTelemetry printed at the end

Example trace hierarchy:

```text
joke_generator.workflow          ← top-level workflow
└── generate_joke.task           ← your task function
    └── chat openai/gpt-oss-120b   ← automatic Groq instrumentation
```

Each span includes metadata such as the model name, token usage, and the prompt/response (when content tracing is enabled).

## How the example works

```python
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

Traceloop.init(app_name="groq_example", disable_batch=True)

@task(name="generate_joke")
def generate_joke():
    # Groq call is traced automatically
    ...

@workflow(name="joke_generator")
def joke_generator():
    generate_joke()
```

See [`sample_app/groq_example.py`](./sample_app/groq_example.py) for the full script.

## Exporting traces (optional)

By default, traces print to your terminal via `ConsoleSpanExporter`.

To send traces to the [Traceloop cloud](https://app.traceloop.com) instead, set in `.env`:

```bash
TRACELOOP_API_KEY=your-traceloop-api-key
```

See the [getting started guide](https://traceloop.com/docs/openllmetry/getting-started-python) for other backends (Datadog, Grafana, etc.).

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Missing Traceloop API key` | Ignore if traces still print to terminal, or set `TRACELOOP_API_KEY` |
| `GROQ_API_KEY` not set | Export the variable: `export GROQ_API_KEY=gsk-...` |
| Model not found (404) | Check [Groq models docs](https://console.groq.com/docs/models) and update `MODEL` in `groq_example.py` |
| `proxies` TypeError from Groq | Known `groq`/`httpx` version mismatch — run with `uv run --with 'groq>=0.18' python sample_app/groq_example.py` |

## More examples

Browse [`sample_app/`](./sample_app/) for other providers and frameworks:

| Category | Examples |
|----------|----------|
| LLM providers | `openai_streaming.py`, `anthropic_joke_example.py`, `cohere_example.py` |
| Local models | `ollama_streaming.py` |
| Frameworks | `langchain_app.py`, `langgraph_example.py`, `crewai_example.py` |
| Vector DBs | `chroma_app.py`, `pinecone_app.py`, `qdrant_app.py` |

## Development commands

From the repository root:

```bash
npx nx run sample-app:lint
npx nx run sample-app:test
```

## Contributing

- [Contributing guide](https://traceloop.com/docs/openllmetry/contributing/overview)
- [Slack community](https://traceloop.com/slack)
