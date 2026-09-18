# OpenTelemetry OCI Generative AI Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-oci-genai/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-oci-genai.svg">
</a>

This library allows tracing calls to the [OCI Generative AI](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm) inference service made with the official [OCI Python SDK](https://github.com/oracle/oci-python-sdk) (`oci.generative_ai_inference.GenerativeAiInferenceClient`).

Instrumented operations:

| SDK method      | `gen_ai.operation.name` | Notes                                                                    |
| --------------- | ----------------------- | ------------------------------------------------------------------------ |
| `chat`          | `chat`                  | `GENERIC`, `COHERE` and `COHEREV2` request formats, streaming and non-streaming |
| `generate_text` | `text_completion`       | Legacy Cohere / Llama runtimes                                           |
| `embed_text`    | `embeddings`            | Emits `gen_ai.embeddings.dimension.count`                                |
| `rerank_text`   | `rerank`                | Emits `gen_ai.oci.rerank.top_n` and document count                        |

Spans follow the OpenTelemetry GenAI semantic conventions (`{operation} {model}` span names, `gen_ai.provider.name`,
`gen_ai.request.*`, `gen_ai.response.*`, `gen_ai.usage.*`, `gen_ai.input.messages` / `gen_ai.output.messages`).
`gen_ai.provider.name` is set to `oracle_cloud.generative_ai`. Both on-demand (`OnDemandServingMode.model_id`) and
dedicated (`DedicatedServingMode.endpoint_id`) serving modes are supported; the serving mode is reported on
`gen_ai.oci.serving_mode`.

For streaming chat requests (`is_stream=True`) the span is completed once the SSE stream returned in
`response.data.events()` has been consumed. Pass `stream_options=StreamOptions(is_include_usage=True)` to receive
token usage in the final stream events.

## Installation

```bash
pip install opentelemetry-instrumentation-oci-genai
```

## Example usage

```python
from opentelemetry.instrumentation.oci_genai import OCIGenAIInstrumentor

OCIGenAIInstrumentor().instrument()
```

## Privacy

**By default, this instrumentation logs prompts, completions, and embeddings to span attributes**. This gives you a clear visibility into how your LLM application is working, and can make it easy to debug and evaluate the quality of the outputs.

However, you may want to disable this logging for privacy reasons, as they may contain highly sensitive data from your users. You may also simply want to reduce the size of your traces.

To disable logging, set the `TRACELOOP_TRACE_CONTENT` environment variable to `false`.

```bash
TRACELOOP_TRACE_CONTENT=false
```
