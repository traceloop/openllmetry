# OpenTelemetry RunPod Instrumentation

This library allows tracing calls to [RunPod Serverless](https://www.runpod.io/) endpoints made with the official
[`runpod`](https://pypi.org/project/runpod/) Python SDK, using OpenTelemetry.

## Installation

```bash
pip install opentelemetry-instrumentation-runpod
```

## Usage

```python
from opentelemetry.instrumentation.runpod import RunpodInstrumentor

RunpodInstrumentor().instrument()

# Now use the RunPod SDK as usual
import runpod

endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

# Blocks until the job finishes and returns its output
output = endpoint.run_sync({"prompt": "Hello, world!"})

# Or submit the job and collect the result through the returned handle
job = endpoint.run({"prompt": "Hello, world!"})
print(job.status())
print(job.output())
```

The asyncio client is instrumented as well:

```python
import runpod
from runpod.http_client import AsyncClientSession

session = AsyncClientSession()
endpoint = runpod.AsyncioEndpoint("YOUR_ENDPOINT_ID", session=session)
job = await endpoint.run({"prompt": "Hello, world!"})
print(await job.output())
```

## Semantic Conventions

This instrumentation follows the OpenTelemetry GenAI semantic conventions where they apply:

| Attribute | Value |
| --- | --- |
| `gen_ai.system` | `runpod` |
| `gen_ai.operation.name` | `run` for the asynchronous submission, `run_sync` for the blocking call |
| `gen_ai.prompt.0.role` / `gen_ai.prompt.0.content` | The request payload, as the SDK normalizes it into `{"input": ...}` |
| `gen_ai.completion.0.role` / `gen_ai.completion.0.content` | The job output, when the call returns it directly |
| `runpod.endpoint_id` | The RunPod Serverless endpoint that handled the request |
| `runpod.job_id` | The id of the submitted job, when the SDK response carries one |

Span names are `runpod.run` and `runpod.run_sync`, both `SpanKind.CLIENT`. Failures set the span status to `ERROR` and
record the exception; successful calls set the span status to `OK`.

Prompt and completion content is only recorded when `TRACELOOP_TRACE_CONTENT` is not `false`.

## Covered entry points

| SDK entry point | Instrumented | Notes |
| --- | --- | --- |
| `runpod.Endpoint.run` | yes | span `runpod.run` |
| `runpod.Endpoint.run_sync` | yes | span `runpod.run_sync` |
| `runpod.AsyncioEndpoint.run` | yes | span `runpod.run` (async) |
| `runpod.AsyncioEndpoint.health`, `purge_queue` | no | operational probes, not serverless inference calls |
| `runpod.Endpoint.health`, `purge_queue` | no | same |
| `runpod.Endpoint.runsync` | n/a | does not exist in the SDK - the blocking call is `run_sync` |
| `Job.output` / `Job.stream` / `Job.status` | no | see below |

### Upstream coverage gaps

`Endpoint.run_sync` posts to `/runsync` and returns the job output in the same request, so a single span covers the whole
call. `Endpoint.run` and `AsyncioEndpoint.run` post to `/run` and return a `Job` handle immediately; the SDK fetches the
result later, in separate requests issued by `Job.output()`, `Job.stream()` or `Job.status()`. Those follow-up requests
are made by the `Job` object through either `requests` or aiohttp rather than through the `Endpoint` classes, so they are
not instrumented here - the returned handle is recorded as `runpod.job_id` instead, and users who want the polling traced
can rely on the HTTP instrumentations (`opentelemetry-instrumentation-requests`) for it.

`runpod.run_sync` (the module-level helper some older examples use) does not exist in the `runpod` 1.x SDK, so there is
nothing to instrument for it.
