# Migration Guide: opentelemetry-semantic-conventions-ai v0.4.x → v0.5.x

This guide covers breaking changes introduced when aligning the `opentelemetry-semantic-conventions-ai`
package with the upstream [OTel GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

---

## 1. Removed constants (previously duplicated upstream)

These `SpanAttributes` constants have been **removed**. They are now part of the official
`opentelemetry-semantic-conventions` package. Import them directly from upstream.

```python
# Before
from opentelemetry.semconv_ai import SpanAttributes
span.set_attribute(SpanAttributes.LLM_SYSTEM, "openai")

# After
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
span.set_attribute(GenAIAttributes.GEN_AI_SYSTEM, "openai")
```

| Removed constant | Upstream replacement |
|---|---|
| `SpanAttributes.LLM_SYSTEM` | `GenAIAttributes.GEN_AI_SYSTEM` |
| `SpanAttributes.LLM_REQUEST_MODEL` | `GenAIAttributes.GEN_AI_REQUEST_MODEL` |
| `SpanAttributes.LLM_REQUEST_MAX_TOKENS` | `GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS` |
| `SpanAttributes.LLM_REQUEST_TEMPERATURE` | `GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE` |
| `SpanAttributes.LLM_REQUEST_TOP_P` | `GenAIAttributes.GEN_AI_REQUEST_TOP_P` |
| `SpanAttributes.LLM_TOP_K` | `GenAIAttributes.GEN_AI_REQUEST_TOP_K` |
| `SpanAttributes.LLM_CHAT_STOP_SEQUENCES` | `GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES` |
| `SpanAttributes.LLM_FREQUENCY_PENALTY` | `GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY` |
| `SpanAttributes.LLM_PRESENCE_PENALTY` | `GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY` |
| `SpanAttributes.LLM_RESPONSE_MODEL` | `GenAIAttributes.GEN_AI_RESPONSE_MODEL` |
| `SpanAttributes.LLM_USAGE_COMPLETION_TOKENS` | `GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS` |
| `SpanAttributes.LLM_USAGE_PROMPT_TOKENS` | `GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS` |
| `SpanAttributes.LLM_TOKEN_TYPE` | `GenAIAttributes.GEN_AI_TOKEN_TYPE` |
| `SpanAttributes.LLM_REQUEST_FUNCTIONS` | `GenAIAttributes.GEN_AI_TOOL_DEFINITIONS` |
| `SpanAttributes.LLM_PROMPTS` | `GenAIAttributes.GEN_AI_PROMPT` |
| `SpanAttributes.LLM_COMPLETIONS` | `GenAIAttributes.GEN_AI_COMPLETION` |
| `SpanAttributes.LLM_OPENAI_RESPONSE_SYSTEM_FINGERPRINT` | `GenAIAttributes.GEN_AI_OPENAI_RESPONSE_SYSTEM_FINGERPRINT` |
| `SpanAttributes.LLM_REQUEST_TYPE` | `GenAIAttributes.GEN_AI_OPERATION_NAME` |

> **Note on `LLM_REQUEST_TYPE`**: The old `LLMRequestTypeValues` enum is replaced by
> `GenAiOperationNameValues` from upstream, or by `GenAICustomOperationName` for
> project-specific operation names.

---

## 2. Renamed constants (stay in `SpanAttributes`, new `GEN_AI_*` prefix)

These constants remain in the `opentelemetry-semantic-conventions-ai` package but their
Python names have been renamed from `LLM_*` to `GEN_AI_*`.

```python
# Before
from opentelemetry.semconv_ai import SpanAttributes
span.set_attribute(SpanAttributes.LLM_IS_STREAMING, True)

# After
from opentelemetry.semconv_ai import SpanAttributes
span.set_attribute(SpanAttributes.GEN_AI_IS_STREAMING, True)
```

| Old name | New name |
|---|---|
| `SpanAttributes.LLM_USAGE_TOTAL_TOKENS` | `SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS` |
| `SpanAttributes.LLM_USER` | `SpanAttributes.GEN_AI_USER` |
| `SpanAttributes.LLM_HEADERS` | `SpanAttributes.GEN_AI_HEADERS` |
| `SpanAttributes.LLM_IS_STREAMING` | `SpanAttributes.GEN_AI_IS_STREAMING` |
| `SpanAttributes.LLM_REQUEST_REPETITION_PENALTY` | `SpanAttributes.GEN_AI_REQUEST_REPETITION_PENALTY` |
| `SpanAttributes.LLM_REQUEST_REASONING_EFFORT` | `SpanAttributes.GEN_AI_REQUEST_REASONING_EFFORT` |
| `SpanAttributes.LLM_REQUEST_REASONING_SUMMARY` | `SpanAttributes.GEN_AI_REQUEST_REASONING_SUMMARY` |
| `SpanAttributes.LLM_RESPONSE_REASONING_EFFORT` | `SpanAttributes.GEN_AI_RESPONSE_REASONING_EFFORT` |
| `SpanAttributes.LLM_RESPONSE_FINISH_REASON` | `SpanAttributes.GEN_AI_RESPONSE_FINISH_REASON` |
| `SpanAttributes.LLM_RESPONSE_STOP_REASON` | `SpanAttributes.GEN_AI_RESPONSE_STOP_REASON` |
| `SpanAttributes.LLM_CONTENT_COMPLETION_CHUNK` | `SpanAttributes.GEN_AI_CONTENT_COMPLETION_CHUNK` |
| `SpanAttributes.LLM_USAGE_REASONING_TOKENS` | `SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS` |
| `SpanAttributes.LLM_USAGE_TOKEN_TYPE` | `SpanAttributes.GEN_AI_USAGE_TOKEN_TYPE` |
| `SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS` | `SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS` ¹ |
| `SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS` | `SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS` ¹ |
| `SpanAttributes.LLM_REQUEST_STRUCTURED_OUTPUT_SCHEMA` | `SpanAttributes.GEN_AI_REQUEST_STRUCTURED_OUTPUT_SCHEMA` |
| `SpanAttributes.LLM_OPENAI_API_BASE` | `SpanAttributes.GEN_AI_OPENAI_API_BASE` |
| `SpanAttributes.LLM_OPENAI_API_VERSION` | `SpanAttributes.GEN_AI_OPENAI_API_VERSION` |
| `SpanAttributes.LLM_OPENAI_API_TYPE` | `SpanAttributes.GEN_AI_OPENAI_API_TYPE` |
| `SpanAttributes.LLM_DECODING_METHOD` | `SpanAttributes.GEN_AI_WATSONX_DECODING_METHOD` |
| `SpanAttributes.LLM_RANDOM_SEED` | `SpanAttributes.GEN_AI_WATSONX_RANDOM_SEED` |
| `SpanAttributes.LLM_MAX_NEW_TOKENS` | `SpanAttributes.GEN_AI_WATSONX_MAX_NEW_TOKENS` |
| `SpanAttributes.LLM_MIN_NEW_TOKENS` | `SpanAttributes.GEN_AI_WATSONX_MIN_NEW_TOKENS` |
| `SpanAttributes.LLM_REPETITION_PENALTY` | `SpanAttributes.GEN_AI_WATSONX_REPETITION_PENALTY` |

> ¹ The string value of these two cache-token attributes **also changed** — see [section 3](#cache-token-attributes).

---

## 3. Changed string values

Some constants kept their Python name but the underlying **string value** changed.

### Cache token attributes

| Python name | Old string value | New string value |
|---|---|---|
| `SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS` | `gen_ai.usage.cache_creation_input_tokens` | `gen_ai.usage.cache_creation.input_tokens` |
| `SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS` | `gen_ai.usage.cache_read_input_tokens` | `gen_ai.usage.cache_read.input_tokens` |

> **Dashboard impact**: Update any Grafana queries, alerts, or OTLP processors that filter on
> these attribute names.

### `GenAISystem` values

All `GenAISystem` enum values now use the OTel spec canonical form (lowercase / dot-separated).

| Enum member | Old value | New value |
|---|---|---|
| `GenAISystem.ANTHROPIC` | `"Anthropic"` | `"anthropic"` |
| `GenAISystem.COHERE` | `"Cohere"` | `"cohere"` |
| `GenAISystem.MISTRALAI` | `"MistralAI"` | `"mistral_ai"` |
| `GenAISystem.OLLAMA` | `"Ollama"` | `"ollama"` |
| `GenAISystem.GROQ` | `"Groq"` | `"groq"` |
| `GenAISystem.ALEPH_ALPHA` | `"AlephAlpha"` | `"aleph_alpha"` |
| `GenAISystem.REPLICATE` | `"Replicate"` | `"replicate"` |
| `GenAISystem.TOGETHER_AI` | `"TogetherAI"` | `"together_ai"` |
| `GenAISystem.WATSONX` | `"Watsonx"` | `"ibm.watsonx.ai"` |
| `GenAISystem.HUGGINGFACE` | `"HuggingFace"` | `"hugging_face"` |
| `GenAISystem.FIREWORKS` | `"Fireworks"` | `"fireworks"` |
| `GenAISystem.AZURE` | `"Azure"` | `"az.ai.openai"` |
| `GenAISystem.AWS` | `"AWS"` | `"aws.bedrock"` |
| `GenAISystem.GOOGLE` | `"Google"` | `"gcp.gen_ai"` |
| `GenAISystem.OPENROUTER` | `"OpenRouter"` | `"openrouter"` |
| `GenAISystem.LANGCHAIN` | `"Langchain"` | `"langchain"` |

> `GenAISystem.OPENAI` (`"openai"`) is unchanged.

> **Dashboard impact**: Update dashboards, alerts, and OTLP processors that filter on
> `gen_ai.system` to use the new lowercase values shown above.

---

## 4. Tool definitions format change

Tool definitions are now encoded as a **single JSON-array attribute** instead of per-field
indexed sub-attributes.

```python
# Before — multiple flat attributes
span.set_attribute("gen_ai.tool.definitions.0.name", "my_tool")
span.set_attribute("gen_ai.tool.definitions.0.description", "Does something")
span.set_attribute("gen_ai.tool.definitions.0.parameters", json.dumps({...}))

# After — one JSON array attribute
import json
tool_defs = [
    {
        "name": "my_tool",
        "description": "Does something",
        "parameters": {...},
    }
]
span.set_attribute(GenAIAttributes.GEN_AI_TOOL_DEFINITIONS, json.dumps(tool_defs))
```

> **Dashboard impact**: Dashboards that expand `gen_ai.tool.definitions.{i}.name` as individual
> attributes will no longer find them. Parse the JSON value of `gen_ai.tool.definitions` instead.

---

## 5. Quickstart: minimal import update

```python
# Before
from opentelemetry.semconv_ai import SpanAttributes

SpanAttributes.LLM_SYSTEM          # removed
SpanAttributes.LLM_REQUEST_MODEL   # removed
SpanAttributes.LLM_REQUEST_TYPE    # removed
SpanAttributes.LLM_IS_STREAMING    # renamed
SpanAttributes.LLM_USAGE_TOTAL_TOKENS  # renamed

# After
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes

GenAIAttributes.GEN_AI_SYSTEM          # upstream
GenAIAttributes.GEN_AI_REQUEST_MODEL   # upstream
GenAIAttributes.GEN_AI_OPERATION_NAME  # upstream
SpanAttributes.GEN_AI_IS_STREAMING     # project semconv (renamed)
SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS  # project semconv (renamed)
```
