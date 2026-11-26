# traceloop-sdk

Traceloop’s Python SDK allows you to easily start monitoring and debugging your LLM execution. Tracing is done in a non-intrusive way, built on top of OpenTelemetry. You can choose to export the traces to Traceloop, or to your existing observability stack.

For agent-to-agent HTTP calls, use the opt-in W3C helpers to carry the current
trace across service boundaries:

```python
from opentelemetry import trace
from traceloop.sdk import inject_trace_context, extract_trace_context

headers = inject_trace_context()
remote_context = extract_trace_context(request.headers)
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("handle-agent-request", context=remote_context):
    handle_request()
```

The helpers use the application's configured global propagator. OpenTelemetry
defaults to W3C Trace Context and Baggage; custom propagators passed through
`Traceloop.init(propagator=...)` remain supported.

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
