# Basic Aleph Alpha completion tracing

This example shows the smallest working setup for tracing an Aleph Alpha completion with OpenLLMetry. It uses the OpenTelemetry console exporter, so no Traceloop account or collector is required: the completed span is printed directly to the terminal.

## Run

From this directory:

```bash
uv run --no-project --with-requirements requirements.txt python main.py
```

Set your Aleph Alpha token first:

```bash
# macOS/Linux
export AA_TOKEN="your-token"

# Windows PowerShell
$env:AA_TOKEN = "your-token"

# Windows cmd.exe
set AA_TOKEN=your-token
```

The default model is `luminous-base`. Override it with `AA_MODEL` when your account uses a different model:

```bash
AA_MODEL="your-model" uv run --no-project --with-requirements requirements.txt python main.py
```

The terminal prints the model response followed by an OpenTelemetry span named `alephalpha.completion`. The span includes the model, request type, and usage attributes. Prompt and completion content follows the instrumentation privacy setting; set `TRACELOOP_TRACE_CONTENT=false` to disable content capture.

## What to look for

1. `AlephAlphaInstrumentor().instrument(...)` enables instrumentation before the client call.
2. `ConsoleSpanExporter` makes the result visible without any external service.
3. The completion response is printed separately from the span, so it is easy to distinguish application output from telemetry.

For production, replace the console exporter with an OTLP exporter and configure a collector.
