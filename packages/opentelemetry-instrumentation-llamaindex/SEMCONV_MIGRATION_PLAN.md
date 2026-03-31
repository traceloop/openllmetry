# LlamaIndex Instrumentation — Semantic Convention Migration Plan

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Target State](#2-target-state)
3. [Refactoring Prerequisites](#3-refactoring-prerequisites)
4. [Phase 1: Unit Tests (Write First)](#4-phase-1-unit-tests-write-first)
5. [Phase 2: Implementation](#5-phase-2-implementation)
6. [Phase 3: Integration Test Updates](#6-phase-3-integration-test-updates)
7. [Dependency & Versioning](#7-dependency--versioning)
8. [Gateway Impact Assessment](#8-gateway-impact-assessment)

---

## 1. Current State Analysis

### 1.1 Package Architecture

The LlamaIndex instrumentation has **two instrumentation paths**:

- **Dispatcher path** (llama-index-core >= 0.10.20): Uses LlamaIndex's built-in dispatcher system
  (`OpenLLMetrySpanHandler` + `OpenLLMetryEventHandler` in `dispatcher_wrapper.py`).
  This is the primary path for modern LlamaIndex.
- **Legacy wrapt path** (older versions): Direct monkey-patching via individual `*_instrumentor.py` files.

### 1.2 Files & Responsibility Map

| File | Responsibility | Semconv Impact |
|------|---------------|----------------|
| `__init__.py` | Orchestrator — picks dispatcher vs legacy path | Low (config wiring only) |
| `dispatcher_wrapper.py` | Core span lifecycle (SpanHolder, SpanHandler, EventHandler) | **HIGH** — sets traceloop.entity.*, suppression keys |
| `span_utils.py` | Sets LLM chat/completion/embedding/rerank attributes on spans | **CRITICAL** — all legacy indexed `gen_ai.prompt.{i}.*` / `gen_ai.completion.{i}.*` attrs |
| `event_emitter.py` | Emits OTel log events (non-legacy mode) | **HIGH** — finish_reason handling, message/choice events |
| `event_models.py` | Dataclasses for MessageEvent/ChoiceEvent | Medium — schema alignment |
| `custom_llm_instrumentor.py` | Wraps CustomLLM.chat/complete | **HIGH** — uses legacy `gen_ai.prompt.{i}.*`, `gen_ai.completion.{i}.*`, `LLM_REQUEST_TYPE`, `GEN_AI_SYSTEM` |
| `utils.py` | Helpers: should_send_prompts, dont_throw, process_request/response | Low-Medium |
| `config.py` | Global config (use_legacy_attributes, event_logger) | Low |
| `base_agent_instrumentor.py` | Wraps AgentRunner.chat/achat | Low (traceloop.* attrs only, no LLM-specific semconv) |
| `base_embedding_instrumentor.py` | Wraps BaseEmbedding.get_query_embedding | Low (traceloop.* attrs only) |
| `base_retriever_instrumentor.py` | Wraps BaseRetriever.retrieve | Low (traceloop.* attrs only) |
| `base_synthesizer_instrumentor.py` | Wraps BaseSynthesizer.synthesize | Low (traceloop.* attrs only) |
| `base_tool_instrumentor.py` | Wraps FunctionTool/QueryEngineTool.call | Low (traceloop.* attrs only) |
| `query_pipeline_instrumentor.py` | Wraps QueryPipeline.run | Low (traceloop.* attrs only) |
| `retriever_query_engine_instrumentor.py` | Wraps RetrieverQueryEngine.query | Low (traceloop.* attrs only) |
| `llamaparse_instrumentor.py` | Wraps LlamaParse methods | None (no LLM semconv) |

### 1.3 Current Semconv Violations

#### 1.3.1 Legacy Indexed Attributes (P1 — must migrate)

**`span_utils.py`** uses the old indexed attribute format throughout:

```python
# Input messages — OLD format
span.set_attribute(f"{GenAIAttributes.GEN_AI_PROMPT}.{idx}.role", message.role.value)
span.set_attribute(f"{GenAIAttributes.GEN_AI_PROMPT}.{idx}.content", message.content)

# Output messages — OLD format
span.set_attribute(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role", response.message.role.value)
span.set_attribute(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content", response.message.content)
```

**Must become** JSON arrays per OTel GenAI semconv v1.40.0+:

```python
# gen_ai.input.messages — JSON array of {role, parts}
# gen_ai.output.messages — JSON array of {role, parts, finish_reason}
```

**`custom_llm_instrumentor.py`** same issue:

```python
span.set_attribute(f"{GenAIAttributes.GEN_AI_PROMPT}.0.user", prompt)
span.set_attribute(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content", response.text)
```

#### 1.3.2 Missing Required Attributes (P1)

- **`gen_ai.operation.name`**: Not set on LlamaIndex-mediated LLM spans in `span_utils.py`.
  Only set via `SpanAttributes.LLM_REQUEST_TYPE` (legacy constant, legacy string value `"chat"`).
- **`gen_ai.provider.name`**: `custom_llm_instrumentor.py` sets `GEN_AI_SYSTEM` to `instance.__class__.__name__`
  (e.g. `"Cohere"`, `"Ollama"`) instead of OTel well-known values.
  `span_utils.py` does not set provider name at all for dispatcher-mediated spans.
- **`gen_ai.response.finish_reasons`**: Not set as a top-level span array attribute.
  Currently only `SpanAttributes.LLM_RESPONSE_FINISH_REASON` (singular, legacy) is set in
  `span_utils.py:set_llm_chat_response_model_attributes` for OpenAI-style responses.

#### 1.3.3 Missing Content Attributes (P2)

- **`gen_ai.input.messages`**: Not set in JSON format anywhere.
- **`gen_ai.output.messages`**: Not set in JSON format anywhere.
- **`gen_ai.system_instructions`**: Not applicable (LlamaIndex passes system messages inline).
- **`gen_ai.tool.definitions`**: Not set.
- **Per-message `finish_reason`**: Not included in output messages.

#### 1.3.4 Legacy Constants Still in Use (P2)

| Current Usage | File | Should Be |
|---------------|------|-----------|
| `SpanAttributes.LLM_REQUEST_TYPE` | `span_utils.py:42`, `custom_llm_instrumentor.py:149` | `GenAIAttributes.GEN_AI_OPERATION_NAME` |
| `GenAIAttributes.GEN_AI_SYSTEM` | `custom_llm_instrumentor.py:148` | `GenAIAttributes.GEN_AI_PROVIDER_NAME` |
| `SpanAttributes.LLM_USAGE_TOTAL_TOKENS` | `span_utils.py:148,151` | `SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS` |
| `SpanAttributes.LLM_RESPONSE_FINISH_REASON` | `span_utils.py:157` | `GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS` (array) |
| `GenAIAttributes.GEN_AI_PROMPT` (indexed) | `span_utils.py`, `custom_llm_instrumentor.py` | `GenAIAttributes.GEN_AI_INPUT_MESSAGES` (JSON) |
| `GenAIAttributes.GEN_AI_COMPLETION` (indexed) | `span_utils.py`, `custom_llm_instrumentor.py` | `GenAIAttributes.GEN_AI_OUTPUT_MESSAGES` (JSON) |
| `GenAIAttributes.GEN_AI_SYSTEM` in event_emitter | `event_emitter.py:37` | `GenAIAttributes.GEN_AI_PROVIDER_NAME` |

#### 1.3.5 Finish Reason Issues (P1)

- `span_utils.py:set_llm_chat_response_model_attributes` only handles OpenAI-style `choices[0].finish_reason`.
  No mapping from provider values to OTel enum (`stop`, `tool_call`, `length`, `content_filter`, `error`).
- `event_emitter.py:emit_chat_response_events` defaults to `"unknown"` — non-compliant.
  Per OTel OutputMessage JSON schema, `finish_reason` must be a string from the FinishReason enum or a custom string.
  Per upstream, fallback should be `""` (empty string), not `"unknown"`.
- No top-level `gen_ai.response.finish_reasons` span array attribute is set.

### 1.4 LlamaIndex-Specific Complexity

LlamaIndex is a **framework that wraps multiple LLM providers**. Key challenges:

1. **Provider-agnostic raw responses**: `response.raw` can be OpenAI, Cohere, Anthropic, or dict format.
   `set_llm_chat_response_model_attributes` already handles OpenAI and Cohere token formats.
   Finish reason extraction must similarly be multi-provider.

2. **Delegation to OpenLLMetry instrumentations**: When `class_name in AVAILABLE_OPENLLMETRY_INSTRUMENTATIONS`
   (currently `["OpenAI"]`), LlamaIndex suppresses its own span and lets the OpenAI instrumentation handle it.
   In this case, the OpenAI instrumentation already emits semconv-compliant attributes.
   **We must NOT double-set attributes for these delegated spans.**

3. **Dispatcher path vs. legacy path**: The dispatcher path uses LlamaIndex's event system
   (`LLMChatStartEvent`, `LLMChatEndEvent`, etc.). The legacy path uses direct `wrapt` wrapping.
   Both must produce identical semconv output.

---

## 2. Target State

After migration, the LlamaIndex instrumentation must:

### 2.1 Span Attributes (all LLM spans)

| Attribute | Where Set | Format |
|-----------|-----------|--------|
| `gen_ai.operation.name` | Request time | `"chat"`, `"text_completion"`, `"embeddings"` |
| `gen_ai.provider.name` | Request time | Detect from instance class or raw response |
| `gen_ai.request.model` | Request time | String (already correct) |
| `gen_ai.request.temperature` | Request time | Float (already correct) |
| `gen_ai.request.max_tokens` | Request time | Int (already correct) |
| `gen_ai.response.model` | Response time | String (already correct) |
| `gen_ai.response.id` | Response time | String (extract from raw) |
| `gen_ai.response.finish_reasons` | Response time | `string[]` — top-level span attribute, mapped values |
| `gen_ai.usage.input_tokens` | Response time | Int (already correct) |
| `gen_ai.usage.output_tokens` | Response time | Int (already correct) |
| `gen_ai.usage.total_tokens` | Response time | Int (migrate constant) |
| `gen_ai.input.messages` | Request time (gated) | JSON string — `[{role, parts}]` |
| `gen_ai.output.messages` | Response time (gated) | JSON string — `[{role, parts, finish_reason}]` |

### 2.2 Message JSON Format

**Input messages** (`gen_ai.input.messages`):

```json
[
  {"role": "system", "parts": [{"type": "text", "content": "You are..."}]},
  {"role": "user", "parts": [{"type": "text", "content": "Hello"}]},
  {"role": "assistant", "parts": [{"type": "text", "content": "Hi"}, {"type": "tool_call", "id": "...", "name": "...", "arguments": {...}}]},
  {"role": "tool", "parts": [{"type": "tool_call_response", "id": "...", "response": "..."}]}
]
```

**Output messages** (`gen_ai.output.messages`):

```json
[
  {
    "role": "assistant",
    "parts": [{"type": "text", "content": "The answer is 42."}],
    "finish_reason": "stop"
  }
]
```

### 2.3 Finish Reason Rules

- **Top-level `gen_ai.response.finish_reasons`**: Set as `string[]`. Omit entirely if no finish reason available. Never fabricate `"stop"`.
- **Per-message `finish_reason`**: Required field in output message JSON. Always a string. Use mapped OTel value or `""` as fallback.
- **Mapping**: Provider-specific → OTel enum. Since LlamaIndex wraps multiple providers, we need a generic mapper that handles OpenAI, Cohere, and unknown values.

---

## 3. Refactoring Prerequisites

Before implementing semconv changes, these code-quality improvements are needed to make the code testable.

### 3.1 `span_utils.py` — Refactoring Needed

#### 🔴 `set_llm_chat_response_model_attributes` (lines 80-158) — SPAGHETTI, 78 lines

**Problem**: One giant function handling model extraction, token usage (3 different formats: OpenAI, Cohere tokens, Cohere billed_units), and finish reason — all with nested `getattr`/`isinstance`/`dict` checks. Impossible to unit-test individual concerns.

**Refactor into**:
- `_extract_model_from_raw(raw) -> Optional[str]`
- `_extract_token_usage(raw) -> TokenUsage` (dataclass with input/output/total)
- `_extract_finish_reasons(raw) -> List[str]` (returns mapped OTel values)
- `_set_response_model_attributes(span, event)` — thin orchestrator calling the above

#### 🔴 `set_llm_chat_response` (lines 55-76) — Needs rewrite for JSON format

**Problem**: Sets indexed `gen_ai.prompt.{i}.*` and `gen_ai.completion.{i}.*`. Must be replaced with
`_set_input_messages` and `_set_output_messages` using JSON format.

**Refactor into**:
- `_build_input_messages(messages) -> List[dict]` — pure function, returns list of `{role, parts}` dicts
- `_build_output_messages(response, finish_reason) -> List[dict]` — pure function, returns list of `{role, parts, finish_reason}` dicts
- `_set_input_messages(span, messages)` — calls builder + sets attribute
- `_set_output_messages(span, response, finish_reason)` — calls builder + sets attribute

#### 🟡 `set_llm_chat_request` (lines 22-33) — Needs rewrite for JSON format

Same indexed attribute issue. Replace with `_set_input_messages`.

### 3.2 `custom_llm_instrumentor.py` — Refactoring Needed

#### 🔴 `_handle_request` (lines 147-171) — Mixed concerns

**Problem**: Sets provider name, operation name, model, and prompts all in one function with different formats for chat vs. completion. Hard to test.

**Refactor into**:
- `_set_request_attributes(span, instance, llm_request_type)` — sets operation.name, provider.name, model, etc.
- `_set_input_messages(span, args, kwargs, llm_request_type)` — handles prompt content

#### 🔴 `_handle_response` (lines 174-186) — Minimal, but wrong format

Must add output message JSON, finish_reasons, token usage.

### 3.3 `event_emitter.py` — Refactoring Needed

#### 🟡 `emit_chat_response_events` (lines 46-59) — Fragile finish_reason extraction

**Problem**: Tries `event.response.raw.get("choices", [{}])[0].get("finish_reason", "unknown")` with a bare
try/except. Must use the shared mapper and handle multiple providers.

### 3.4 `dispatcher_wrapper.py` — Moderate refactoring

#### 🟡 `SpanHolder.update_span_for_event` dispatchers (lines 120-155)

These delegate to `span_utils.py` functions. Once span_utils is migrated, these mostly stay the same
but need to pass through the new parameters (e.g., finish_reason for output messages).

#### 🟡 `OpenLLMetrySpanHandler.new_span` (lines 166-246) — Long but structured

80 lines. Acceptable complexity, but the `AVAILABLE_OPENLLMETRY_INSTRUMENTATIONS` delegation logic
should be clearly documented. No structural refactor needed.

---

## 4. Phase 1: Unit Tests (Write First)

All new unit tests go in `tests/test_semconv_migration.py`. Use `unittest.mock.MagicMock` for spans
and `unittest.mock.patch` for `should_send_prompts`. Follow the pattern in `test_none_content_fix.py`.

### 4.1 New Test File: `tests/test_semconv_migration.py`

#### 4.1.1 Message Building Tests

```
class TestBuildInputMessages:
    test_single_user_message
    test_multiple_messages_with_roles
    test_system_message_inline
    test_message_with_none_content
    test_message_with_empty_content
    test_tool_role_message
    test_assistant_message_with_tool_calls  # tool_call parts from additional_kwargs
    test_tool_call_response_round_trip  # tool role message with tool_call_response part
    test_message_order_preserved
    test_multimodal_content_list  # content as list of blocks (text + image)
    test_image_url_mapped_to_uri_part  # image URL → {"type": "uri", "modality": "image", ...}
    test_image_base64_mapped_to_blob_part  # base64 image → {"type": "blob", "modality": "image", ...}
    test_mixed_text_and_image_content  # list with text + image blocks

class TestBuildOutputMessages:
    test_single_assistant_response
    test_response_with_none_content
    test_response_with_finish_reason_stop
    test_response_with_finish_reason_tool_call
    test_response_with_finish_reason_length
    test_response_with_none_finish_reason  # should use "" fallback
    test_response_with_unknown_finish_reason  # pass-through
    test_response_with_tool_call_parts  # assistant response containing tool_call parts
```

#### 4.1.2 Finish Reason Tests

```
class TestMapFinishReason:
    # Generic mapper for LlamaIndex (handles OpenAI + Cohere + unknown)
    test_openai_stop -> "stop"
    test_openai_tool_calls -> "tool_call"
    test_openai_function_call -> "tool_call"
    test_openai_length -> "length"
    test_openai_content_filter -> "content_filter"
    test_cohere_COMPLETE -> "stop"
    test_cohere_MAX_TOKENS -> "length"
    test_cohere_ERROR -> "error"
    test_cohere_ERROR_TOXIC -> "content_filter"
    test_none_returns_none  # for top-level attr: omit
    test_empty_string_returns_none  # empty string treated as missing
    test_unknown_passes_through  # unmapped values pass through as-is (documented behavior)

class TestFinishReasonTopLevel:
    test_set_finish_reasons_array_on_span
    test_omit_when_none
    test_omit_when_empty_list
    test_multiple_choices  # if applicable
    test_not_gated_by_should_send_prompts  # P1: finish_reasons is Recommended, NOT Opt-In

class TestFinishReasonPerMessage:
    test_always_present_in_output_message_json
    test_mapped_value_used
    test_fallback_empty_string_when_none
```

#### 4.1.3 Token Usage Extraction Tests

```
class TestExtractTokenUsage:
    test_openai_format  # usage.completion_tokens, usage.prompt_tokens
    test_openai_format_dict
    test_cohere_meta_tokens_format
    test_cohere_meta_billed_units_format
    test_no_usage_returns_none
    test_partial_usage  # only input_tokens present
```

#### 4.1.4 Model & Response ID Extraction Tests

```
class TestExtractModelFromRaw:
    test_object_with_model_attr
    test_dict_with_model_key
    test_no_model_returns_none

class TestExtractResponseId:
    test_object_with_id_attr
    test_dict_with_id_key
    test_no_id_returns_none
```

#### 4.1.5 Set Input/Output Messages Integration Tests (with mock span)

```
class TestSetInputMessages:
    test_sets_gen_ai_input_messages_json
    test_json_format_matches_otel_schema
    test_gated_by_should_send_prompts
    test_skips_when_span_not_recording
    test_handles_none_messages

class TestSetOutputMessages:
    test_sets_gen_ai_output_messages_json
    test_json_format_matches_otel_schema
    test_includes_finish_reason_per_message
    test_gated_by_should_send_prompts
    test_skips_when_span_not_recording
    test_streaming_produces_same_format_as_non_streaming  # verify identical JSON schema
```

#### 4.1.6 Attribute Name Tests

```
class TestAttributeNames:
    test_operation_name_set_as_gen_ai_operation_name
    test_provider_name_set_as_gen_ai_provider_name  # not gen_ai.system
    test_total_tokens_uses_gen_ai_usage_total_tokens  # not llm.usage.total_tokens
    test_finish_reasons_uses_gen_ai_response_finish_reasons  # array, not singular
```

#### 4.1.7 Provider Name Detection Tests

```
class TestDetectProviderName:
    # Must map to OTel well-known values (see semconv registry)
    test_openai_class -> "openai"
    test_cohere_class -> "cohere"
    test_anthropic_class -> "anthropic"
    test_groq_class -> "groq"
    test_mistralai_class -> "mistral_ai"
    test_bedrock_class -> "aws.bedrock"
    test_gemini_class -> "gcp.gemini"
    test_ollama_class -> "ollama"  # not in OTel well-known, pass through lowercase
    test_custom_llm_class -> lowercase class name as fallback
    test_none_instance -> None
    test_from_model_dict_class_name  # dispatcher path: extract from event.model_dict
    test_from_span_handler_instance  # dispatcher path: extract from new_span(instance=...)
```

### 4.2 New Test File: `tests/test_custom_llm_semconv.py`

```
class TestCustomLLMHandleRequest:
    test_sets_operation_name_chat
    test_sets_operation_name_completion
    test_sets_provider_name  # not class name
    test_sets_input_messages_json_for_completion
    test_sets_input_messages_json_for_chat
    test_gated_by_should_send_prompts

class TestCustomLLMHandleResponse:
    test_sets_output_messages_json
    test_sets_response_model
    test_sets_finish_reasons_array
```

### 4.3 New Test File: `tests/test_event_emitter_semconv.py`

```
class TestEmitChatResponseEvents:
    test_finish_reason_mapped_to_otel_value
    test_finish_reason_fallback_empty_string
    test_provider_name_attribute  # gen_ai.provider.name, not gen_ai.system

class TestEmitChatMessageEvents:
    # Existing behavior, ensure no regression
    test_message_events_emitted_for_each_input_message
```

### 4.4 Updates to Existing Test Files

#### `tests/test_none_content_fix.py`

- Update assertions to check new JSON format (`gen_ai.input.messages`, `gen_ai.output.messages`)
  instead of indexed `gen_ai.completion.{i}.*` attributes.
- Keep the None-content edge case tests — they're valuable.

#### `tests/test_agents.py`

- Update `test_agents_and_tools`: Verify `gen_ai.response.finish_reasons` on LLM spans.
- Update `test_agent_with_multiple_tools`: Verify Cohere spans set `gen_ai.operation.name`,
  `gen_ai.response.finish_reasons`, and use `gen_ai.input.messages`/`gen_ai.output.messages` JSON.
- Remove assertions on `gen_ai.prompt.*` and `gen_ai.completion.*` indexed attributes.
- Replace `SpanAttributes.LLM_USAGE_TOTAL_TOKENS` assertions with `SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS`.

#### `tests/test_chroma_vector_store.py`

- Update assertions for `gen_ai.input.messages` / `gen_ai.output.messages` JSON format.
- Replace `SpanAttributes.LLM_USAGE_TOTAL_TOKENS` with `SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS`.

#### `tests/test_structured_llm.py`

- Add assertion for `gen_ai.operation.name` == `"chat"`.
- No other changes needed (model/temperature assertions already use correct constants).

---

## 5. Phase 2: Implementation

### 5.1 Step 1: Create `_message_utils.py` — Pure message-building functions

New file: `opentelemetry/instrumentation/llamaindex/_message_utils.py`

Contains all pure functions for building semconv-compliant message JSON.
No side effects, no span interaction — maximally testable.

```python
"""Pure functions for building OTel GenAI semconv-compliant message JSON."""

import json
from typing import Any, Dict, List, Optional

# Finish reason mapping: covers OpenAI, Cohere, and passes through unknown values.
# Values already matching OTel enum (stop, length, content_filter, error) pass through
# unchanged via the dict.get() fallback — this is intentional.
_FINISH_REASON_MAP = {
    # OpenAI
    "tool_calls": "tool_call",       # OpenAI plural → OTel singular
    "function_call": "tool_call",    # OpenAI legacy → OTel
    # Cohere (surfaced through LlamaIndex response.raw)
    "COMPLETE": "stop",
    "MAX_TOKENS": "length",
    "ERROR": "error",
    "ERROR_TOXIC": "content_filter",
    # Anthropic (if LlamaIndex surfaces raw Anthropic responses)
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_call",
    "max_tokens": "length",
}


def map_finish_reason(reason: Optional[str]) -> Optional[str]:
    """Map provider finish_reason to OTel enum value.

    Contract:
    - Returns None if reason is None or empty (callers for top-level attr should omit).
    - Returns mapped OTel value or pass-through for unmapped values.
    - For per-message finish_reason (required by OutputMessage schema), callers MUST
      apply a fallback: `map_finish_reason(reason) or ""`.
    """
    if not reason:
        return None
    return _FINISH_REASON_MAP.get(reason, reason)


def _parse_arguments(arguments) -> Any:
    """Parse tool call arguments to an object. Best-effort json.loads with fallback."""
    if arguments is None:
        return None
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return arguments
    return arguments


def _content_to_parts(content) -> List[Dict]:
    """Convert LlamaIndex message content to OTel parts array.

    Handles:
    - str/None → single TextPart or empty
    - list of content blocks → map each by type (text, image_url, etc.)
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "content": content}] if content else []
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append({"type": "text", "content": block})
            elif isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "text":
                    parts.append({"type": "text", "content": block.get("text", "")})
                elif block_type == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    parts.append({"type": "uri", "modality": "image", "uri": url})
                elif block_type == "image":
                    # Handle base64 or URL source
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        parts.append({
                            "type": "blob", "modality": "image",
                            "mime_type": source.get("media_type", ""),
                            "content": source.get("data", ""),
                        })
                    elif source.get("type") == "url":
                        parts.append({
                            "type": "uri", "modality": "image",
                            "uri": source.get("url", ""),
                        })
                else:
                    # Fallback: treat as text if it has content
                    if "text" in block:
                        parts.append({"type": "text", "content": block["text"]})
                    elif "content" in block:
                        parts.append({"type": "text", "content": str(block["content"])})
            else:
                # Fallback for non-dict, non-str items
                parts.append({"type": "text", "content": str(block)})
        return parts
    # Fallback: stringify
    return [{"type": "text", "content": str(content)}]


def _extract_tool_calls_from_message(msg) -> List[Dict]:
    """Extract tool_call parts from a LlamaIndex ChatMessage's additional_kwargs."""
    tool_calls = getattr(msg, "additional_kwargs", {}).get("tool_calls", [])
    parts = []
    for tc in tool_calls:
        func = tc.get("function", {}) if isinstance(tc, dict) else {}
        parts.append({
            "type": "tool_call",
            "id": tc.get("id") if isinstance(tc, dict) else None,
            "name": func.get("name", ""),
            "arguments": _parse_arguments(func.get("arguments")),
        })
    return parts


def build_input_messages(messages) -> List[Dict]:
    """Build OTel-compliant input messages from LlamaIndex ChatMessage list.

    Handles multimodal content, tool calls in assistant messages, and
    tool_call_response parts in tool-role messages.
    """
    result = []
    for msg in messages:
        role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
        parts = _content_to_parts(msg.content)

        # Extract tool_call parts from assistant messages
        if role == "assistant":
            parts.extend(_extract_tool_calls_from_message(msg))

        # For tool-role messages, wrap content as tool_call_response
        if role == "tool":
            tool_call_id = getattr(msg, "additional_kwargs", {}).get("tool_call_id")
            if tool_call_id and parts:
                # Convert text parts to tool_call_response
                response_content = parts[0].get("content", "") if parts else ""
                parts = [{
                    "type": "tool_call_response",
                    "id": tool_call_id,
                    "response": response_content,
                }]

        result.append({"role": role, "parts": parts})
    return result


def build_output_message(response_message, finish_reason: Optional[str] = None) -> Dict:
    """Build a single OTel-compliant output message from a LlamaIndex response message."""
    role = response_message.role.value if hasattr(response_message.role, 'value') else "assistant"
    parts = _content_to_parts(response_message.content)

    # Extract tool_call parts if present
    tool_call_parts = _extract_tool_calls_from_message(response_message)
    parts.extend(tool_call_parts)

    # Per OTel OutputMessage schema, finish_reason is required and must be a string
    fr = map_finish_reason(finish_reason) or ""

    return {"role": role, "parts": parts, "finish_reason": fr}


def build_completion_output_message(text: str, finish_reason: Optional[str] = None) -> Dict:
    """Build output message for text completion responses."""
    fr = map_finish_reason(finish_reason) or ""
    parts = [{"type": "text", "content": text}] if text else []
    return {"role": "assistant", "parts": parts, "finish_reason": fr}
```

### 5.2 Step 2: Create `_response_utils.py` — Response data extraction

New file: `opentelemetry/instrumentation/llamaindex/_response_utils.py`

```python
"""Utilities for extracting structured data from LlamaIndex raw responses."""

from dataclasses import dataclass
from typing import Any, List, Optional

from ._message_utils import map_finish_reason


@dataclass
class TokenUsage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


# Map LlamaIndex LLM class names to OTel well-known provider values.
# See: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
_PROVIDER_MAP = {
    "OpenAI": "openai",
    "AzureOpenAI": "azure.ai.openai",
    "Anthropic": "anthropic",
    "Cohere": "cohere",
    "Groq": "groq",
    "MistralAI": "mistral_ai",
    "Bedrock": "aws.bedrock",
    "Gemini": "gcp.gemini",
    "VertexAI": "gcp.vertex_ai",
    "DeepSeek": "deepseek",
    "Perplexity": "perplexity",
    # Ollama not in OTel well-known values — pass through lowercase
}


def detect_provider_name(instance_or_class_name) -> Optional[str]:
    """Detect OTel provider name from a LlamaIndex LLM instance or class name string.

    Supports two call patterns:
    - instance (object): extracts class name from instance.__class__.__name__
    - class_name (str): e.g. from event.model_dict.get("class_name")

    Returns OTel well-known value if available, otherwise lowercase class name.
    Returns None if input is None.
    """
    if instance_or_class_name is None:
        return None
    if isinstance(instance_or_class_name, str):
        class_name = instance_or_class_name
    else:
        class_name = instance_or_class_name.__class__.__name__

    if class_name in _PROVIDER_MAP:
        return _PROVIDER_MAP[class_name]

    # Fallback: lowercase class name (e.g. "Ollama" → "ollama")
    return class_name.lower()


def extract_model_from_raw(raw) -> Optional[str]:
    """Extract model name from raw LLM response (object or dict)."""
    if hasattr(raw, "model"):
        return raw.model
    if isinstance(raw, dict):
        return raw.get("model")
    return None


def extract_response_id(raw) -> Optional[str]:
    """Extract response ID from raw LLM response (object or dict)."""
    if hasattr(raw, "id"):
        return raw.id
    if isinstance(raw, dict):
        return raw.get("id")
    return None


def extract_token_usage(raw) -> TokenUsage:
    """Extract token usage from raw response. Handles OpenAI, Cohere, and dict formats."""
    ...  # Implementation per existing span_utils.py lines 98-151


def extract_finish_reasons(raw) -> List[str]:
    """Extract and map finish reasons from raw LLM response.

    Handles multiple provider formats:
    - OpenAI: raw.choices[i].finish_reason
    - Cohere: raw.finish_reason (str) or raw.meta.finish_reason
    - Anthropic: raw.stop_reason
    - Dict: raw["choices"][i]["finish_reason"]

    Returns list of mapped OTel finish reason strings.
    Returns empty list if no finish reason found (caller should omit top-level attr).
    """
    reasons = []

    # Try OpenAI format: choices[].finish_reason
    choices = getattr(raw, "choices", None)
    if choices is None and isinstance(raw, dict):
        choices = raw.get("choices")
    if choices:
        for choice in choices:
            fr = getattr(choice, "finish_reason", None)
            if fr is None and isinstance(choice, dict):
                fr = choice.get("finish_reason")
            mapped = map_finish_reason(fr)
            if mapped:
                reasons.append(mapped)
        if reasons:
            return reasons

    # Try Anthropic format: stop_reason
    stop_reason = getattr(raw, "stop_reason", None)
    if stop_reason is None and isinstance(raw, dict):
        stop_reason = raw.get("stop_reason")
    if stop_reason:
        mapped = map_finish_reason(stop_reason)
        if mapped:
            return [mapped]

    # Try Cohere format: finish_reason (direct attr or in meta)
    fr = getattr(raw, "finish_reason", None)
    if fr is None and isinstance(raw, dict):
        fr = raw.get("finish_reason")
    if fr:
        mapped = map_finish_reason(fr)
        if mapped:
            return [mapped]

    return reasons
```

### 5.3 Step 3: Migrate `span_utils.py`

1. Replace `set_llm_chat_request` to use `_message_utils.build_input_messages` + set `gen_ai.input.messages` JSON.
2. Replace `set_llm_chat_response` to use `_message_utils.build_output_message` + set `gen_ai.output.messages` JSON.
   - Pass finish_reason from `_response_utils.extract_finish_reasons(event.response.raw)` to `build_output_message`.
3. Rewrite `set_llm_chat_response_model_attributes`:
   - Use `_response_utils.extract_*` functions.
   - Set `gen_ai.response.finish_reasons` as top-level array attribute.
   - **CRITICAL: `gen_ai.response.finish_reasons` must NOT be gated by `should_send_prompts()`**.
     It is a Recommended attribute (metadata), not Opt-In (content). Set it unconditionally
     when available. Spec ref: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
   - Replace `SpanAttributes.LLM_USAGE_TOTAL_TOKENS` with `SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS`.
4. Replace `set_llm_chat_request_model_attributes`:
   - Set `gen_ai.operation.name` = `"chat"`.
   - Detect and set `gen_ai.provider.name` using `_response_utils.detect_provider_name()`.
   - **Provider detection in dispatcher path**: Extract class name from `event.model_dict.get("class_name")`
     (LlamaIndex stores the LLM class name in model_dict). If not available, the `new_span` handler in
     `dispatcher_wrapper.py` receives `instance` — extract from `instance.__class__.__name__` and store
     on the `SpanHolder` so it's available at event time.
5. Update `set_llm_predict_response`:
   - Use `_message_utils.build_completion_output_message` for output messages JSON.
   - Set `gen_ai.operation.name` = `"text_completion"`.
   - Set `gen_ai.response.finish_reasons` if available from response (may be empty — use `""` fallback
     for per-message finish_reason since LLMPredictEndEvent may not carry finish_reason).
   - Set `gen_ai.provider.name` if detectable.

### 5.4 Step 4: Migrate `custom_llm_instrumentor.py`

1. Replace `GEN_AI_SYSTEM` with `GEN_AI_PROVIDER_NAME`, use `detect_provider_name(instance)`.
   - For truly custom LLMs, `detect_provider_name` returns lowercase class name as fallback.
2. Replace `SpanAttributes.LLM_REQUEST_TYPE` with `GenAIAttributes.GEN_AI_OPERATION_NAME`.
   - `LLMRequestTypeValues.CHAT` → `"chat"`, `LLMRequestTypeValues.COMPLETION` → `"text_completion"`.
3. Replace indexed prompt/completion attrs with JSON `gen_ai.input.messages`/`gen_ai.output.messages`.
   - For completion: wrap prompt string in `build_input_messages` equivalent (single user message).
   - For chat: use `build_input_messages(args[0])` if args contain ChatMessage list.
4. Add `gen_ai.response.finish_reasons` to `_handle_response`:
   - CustomLLM responses (`CompletionResponse`, `ChatResponse`) may not carry `raw` with finish_reason.
   - Try `response.raw` if available, fall back to omitting the attribute.
   - **Not gated by `should_send_prompts()`** — set unconditionally when available.

### 5.5 Step 5: Migrate `event_emitter.py`

1. Replace `GEN_AI_SYSTEM` with `GEN_AI_PROVIDER_NAME` in `EVENT_ATTRIBUTES`.
2. Update `emit_chat_response_events`:
   - Use shared `map_finish_reason` from `_message_utils`.
   - Default to `""` instead of `"unknown"` when finish_reason unavailable.

### 5.6 Step 6: Migrate `dispatcher_wrapper.py`

1. Update `SpanHolder.update_span_for_event` handlers to pass new data through.
2. **Provider name on SpanHolder**: Add `provider_name: Optional[str]` field to `SpanHolder`.
   In `OpenLLMetrySpanHandler.new_span`, extract and store provider name:
   ```python
   # In new_span(), after extracting class_name:
   provider_name = detect_provider_name(class_name)  # or from instance
   # Store on SpanHolder for use by event handlers
   ```
   Pass `provider_name` to `set_llm_chat_request_model_attributes` in event handlers.
3. **Streaming finish_reason**: Verify that `LLMChatEndEvent` carries `response.raw` with finish_reason
   even in streaming mode. LlamaIndex fires `LLMChatEndEvent` at stream completion with the
   accumulated response — `set_llm_chat_response_model_attributes` should extract finish_reason
   from it identically to non-streaming. If `response.raw` is `None` in streaming mode,
   `extract_finish_reasons` returns `[]` and the top-level attr is omitted (correct per spec).
4. **Error paths**: The `dont_throw` decorator in `span_utils.py` catches all exceptions silently.
   If attribute-setting fails partway (e.g., input messages set but output messages fail), the span
   will have partial attributes. This is acceptable — the `dont_throw` pattern is intentional to avoid
   breaking the user's application. Add a test verifying partial attribute behavior under exception.

### 5.7 Step 7: Update `event_models.py`

1. Update `ChoiceEvent.finish_reason` default from `"unknown"` to `""`.

### 5.8 Code Path Completeness Matrix

All paths below must set identical semconv attributes in identical format. Verify each during implementation.

| # | Code Path | `gen_ai.operation.name` | `gen_ai.provider.name` | `gen_ai.input.messages` | `gen_ai.output.messages` | `gen_ai.response.finish_reasons` | Notes |
|---|-----------|------------------------|----------------------|------------------------|-------------------------|--------------------------------|-------|
| 1 | Dispatcher: LLM chat non-streaming | ✅ `set_llm_chat_request_model_attributes` | ✅ via `detect_provider_name` (§5.3.4) | ✅ `set_llm_chat_request` (gated) | ✅ `set_llm_chat_response` (gated) | ✅ `set_llm_chat_response_model_attributes` (**ungated**) | Primary path |
| 2 | Dispatcher: LLM chat streaming | Same as #1 | Same as #1 | Same as #1 | Same as #1 (via `LLMChatEndEvent`) | Same as #1 (via `LLMChatEndEvent`) | Verify `response.raw` available in streaming `LLMChatEndEvent` |
| 3 | Dispatcher: LLM predict | ✅ `set_llm_predict_response` → `"text_completion"` | ✅ if detectable (§5.3.5) | N/A (predict uses prompt string) | ✅ `set_llm_predict_response` (gated) | ✅ if available, else omit | `LLMPredictEndEvent` may lack raw response |
| 4 | Dispatcher: Embedding | `"embeddings"` (if added) | ✅ if detectable | N/A | N/A | N/A | Low priority — no chat content |
| 5 | Dispatcher: Rerank | N/A | N/A | N/A | N/A | N/A | No LLM semconv needed |
| 6 | CustomLLM: chat sync | ✅ `_handle_request` → `"chat"` | ✅ `detect_provider_name(instance)` | ✅ `_handle_request` (gated) | ✅ `_handle_response` (gated) | ✅ `_handle_response` (**ungated**) | Via `chat_wrapper` |
| 7 | CustomLLM: chat async | Same as #6 | Same as #6 | Same as #6 | Same as #6 | Same as #6 | Via `achat_wrapper` — shared helpers |
| 8 | CustomLLM: complete sync | ✅ `_handle_request` → `"text_completion"` | ✅ `detect_provider_name(instance)` | ✅ (single user msg, gated) | ✅ `_handle_response` (gated) | ✅ `_handle_response` (**ungated**) | Via `complete_wrapper` |
| 9 | CustomLLM: complete async | Same as #8 | Same as #8 | Same as #8 | Same as #8 | Same as #8 | Via `acomplete_wrapper` — shared helpers |
| 10 | Error/exception paths | Partial (set before error) | Partial (set before error) | Partial (set at request time) | ❌ May be missing | ❌ May be missing | `dont_throw` swallows — acceptable, add test |
| 11 | Delegated OpenAI spans | N/A (suppressed) | N/A (suppressed) | N/A | N/A | N/A | OpenAI instrumentation handles these |

**Key rule**: `gen_ai.response.finish_reasons` is **never** gated by `should_send_prompts()` in any path.

---

## 6. Phase 3: Integration Test Updates

After implementation, update VCR-based integration tests:

1. **Re-record cassettes** for tests that assert on new attribute names/formats.
2. **Update assertions** per Phase 1 §4.4 above.
3. **Add new integration test scenarios** if coverage gaps remain:
   - Streaming response test (if not covered).
   - Error path test (ensure attributes still set on exception).

---

## 7. Dependency & Versioning

### 7.1 `opentelemetry-semantic-conventions-ai`

Current range: `>=0.5.1,<0.6.0`

**Check**: Does `GEN_AI_USAGE_TOTAL_TOKENS` exist in `0.5.1`? Yes — it's in `SpanAttributes`.
No bump needed unless we add new constants.

### 7.2 `opentelemetry-semantic-conventions` (upstream)

Current range: `>=0.59b0`

**Check**: `GenAIAttributes.GEN_AI_PROVIDER_NAME` requires `>=0.50b0` (semconv v1.33.0).
`GenAIAttributes.GEN_AI_INPUT_MESSAGES` requires `>=0.50b0`.
Current `>=0.59b0` is sufficient.

### 7.3 Dependency Compatibility Gate

Before merge, verify:
```bash
pip install --dry-run opentelemetry-instrumentation-llamaindex opentelemetry-instrumentation-anthropic opentelemetry-instrumentation-openai traceloop-sdk
```

---

## 8. Gateway Impact Assessment

**SDK should follow OTel spec. Gateway concerns do NOT block this migration.**

| Change | Gateway Impact |
|--------|---------------|
| `gen_ai.prompt.{i}.*` → `gen_ai.input.messages` JSON | Gateway `buildMessagesJSON()` produces its own JSON from old indexed attrs. New SDK emits ready-made JSON — no conflict, both formats land correctly. |
| `gen_ai.completion.{i}.*` → `gen_ai.output.messages` JSON | Same as above. |
| `llm.request.type` → `gen_ai.operation.name` | Gateway `exactRenames` handles this — no conflict. |
| `gen_ai.system` → `gen_ai.provider.name` | New attribute name, no gateway rename needed (pass-through). |
| `llm.response.finish_reason` → `gen_ai.response.finish_reasons` | Gateway has singular→plural rename. New SDK emits plural directly — no conflict. |
| Per-message `finish_reason` in JSON | Gateway `buildMessagesJSON()` defaults to `"stop"` for old SDKs. New SDK uses mapped value or `""`. Silent data divergence for old-vs-new SDK spans, but acceptable. |

---

## Appendix A: Test Coverage Matrix

| Component | Unit Tests | Integration Tests |
|-----------|-----------|-------------------|
| `_message_utils.build_input_messages` | ✅ test_semconv_migration.py | ✅ test_agents.py, test_chroma_vector_store.py |
| `_message_utils.build_output_message` | ✅ test_semconv_migration.py | ✅ test_agents.py, test_chroma_vector_store.py |
| `_message_utils.map_finish_reason` | ✅ test_semconv_migration.py | ✅ test_agents.py |
| `_response_utils.extract_token_usage` | ✅ test_semconv_migration.py | ✅ test_agents.py (OpenAI + Cohere) |
| `_response_utils.extract_finish_reasons` | ✅ test_semconv_migration.py | ✅ test_agents.py |
| `_response_utils.detect_provider_name` | ✅ test_semconv_migration.py | ✅ test_agents.py |
| `span_utils.set_llm_chat_request` (migrated) | ✅ test_semconv_migration.py | ✅ test_agents.py |
| `span_utils.set_llm_chat_response` (migrated) | ✅ test_semconv_migration.py, test_none_content_fix.py | ✅ test_agents.py |
| `span_utils.set_llm_chat_response_model_attributes` | ✅ test_semconv_migration.py | ✅ test_agents.py |
| `custom_llm_instrumentor._handle_request` | ✅ test_custom_llm_semconv.py | — |
| `custom_llm_instrumentor._handle_response` | ✅ test_custom_llm_semconv.py | — |
| `event_emitter.emit_chat_response_events` | ✅ test_event_emitter_semconv.py | ✅ test_agents.py (non-legacy mode) |
| `event_emitter.EVENT_ATTRIBUTES` | ✅ test_event_emitter_semconv.py | — |
| None content edge case | ✅ test_none_content_fix.py (updated) | — |
| StructuredLLM model attributes | — | ✅ test_structured_llm.py |
| Streaming response | — | ✅ test_agents.py (if streaming cassette added) |
| Delegated OpenAI spans (no double-set) | — | ✅ test_chroma_vector_store.py, test_agents.py |

## Appendix B: File Change Summary

| File | Action |
|------|--------|
| `_message_utils.py` | **NEW** — Pure message/finish_reason builders |
| `_response_utils.py` | **NEW** — Response data extraction utilities |
| `span_utils.py` | **HEAVY EDIT** — Rewrite all attribute-setting functions |
| `custom_llm_instrumentor.py` | **MODERATE EDIT** — Migrate constants + message format |
| `event_emitter.py` | **MODERATE EDIT** — Provider name + finish_reason fix |
| `event_models.py` | **MINOR EDIT** — Default finish_reason `"unknown"` → `""` |
| `dispatcher_wrapper.py` | **MODERATE EDIT** — Provider name on SpanHolder, streaming finish_reason, pass-through updates |
| `utils.py` | **NO CHANGE** |
| `config.py` | **NO CHANGE** |
| `__init__.py` | **NO CHANGE** |
| `base_agent_instrumentor.py` | **NO CHANGE** (traceloop.* attrs only) |
| `base_embedding_instrumentor.py` | **NO CHANGE** |
| `base_retriever_instrumentor.py` | **NO CHANGE** |
| `base_synthesizer_instrumentor.py` | **NO CHANGE** |
| `base_tool_instrumentor.py` | **NO CHANGE** |
| `query_pipeline_instrumentor.py` | **NO CHANGE** |
| `retriever_query_engine_instrumentor.py` | **NO CHANGE** |
| `llamaparse_instrumentor.py` | **NO CHANGE** |
| `tests/test_semconv_migration.py` | **NEW** — Comprehensive unit tests |
| `tests/test_custom_llm_semconv.py` | **NEW** — CustomLLM semconv unit tests |
| `tests/test_event_emitter_semconv.py` | **NEW** — Event emitter semconv tests |
| `tests/test_none_content_fix.py` | **EDIT** — Update for JSON format |
| `tests/test_agents.py` | **EDIT** — Update assertions |
| `tests/test_chroma_vector_store.py` | **EDIT** — Update assertions |
| `tests/test_structured_llm.py` | **MINOR EDIT** — Add operation.name assertion |

---

## Appendix C: Semconv Review (openllmetry-semconv-review checklist)

Applied against this plan per OTel GenAI semconv spec.

### Checklist § 0 — Verify Against Latest OTel Semconv

- [x] Plan references semconv v1.40.0+ message schema — **PASS**
- [x] `gen_ai.usage.total_tokens` checked — remains custom in semconv-ai, no upstream equivalent yet — **PASS**
- [ ] **P2 — GAP**: Plan does not verify `gen_ai.is_streaming`, `gen_ai.usage.reasoning_tokens` against latest spec. These are not used by LlamaIndex today, but plan should note that if streaming detection is added, `gen_ai.is_streaming` should be checked for upstream promotion.

### Checklist § 1 — Message Structure

- [x] Input messages use `gen_ai.input.messages` as JSON `[{role, parts}]` — **PASS** (§5.1)
- [x] Output messages use `gen_ai.output.messages` as JSON `[{role, parts, finish_reason}]` — **PASS** (§5.1)
- [x] `gen_ai.system_instructions` correctly identified as N/A (LlamaIndex inline) — **PASS** (§1.3.3)
- [x] Text parts use `{"type": "text", "content": "..."}` — **PASS** (§5.1 code sample)
- [x] **P2 — RESOLVED: Multimodal content**. Added `_content_to_parts()` to §5.1 handling str/list/image blocks → BlobPart/UriPart. Added 4 multimodal tests to §4.1.1.
- [x] **P2 — RESOLVED: Tool call parts in assistant messages**. Added `_extract_tool_calls_from_message()` to §5.1, extracts from `additional_kwargs`. Added `test_tool_call_response_round_trip` to §4.1.1.
- [x] **P3 — RESOLVED: `arguments` parsing**. Added `_parse_arguments()` helper with `json.loads` + fallback to §5.1.
- [x] Message order preserved — **PASS** (plan iterates messages in order)

### Checklist § 2 — Roles

- [x] Standard OTel roles used (`system`, `user`, `assistant`, `tool`) — **PASS**
- [x] LlamaIndex `MessageRole` enum values map cleanly to OTel roles — **PASS**

### Checklist § 3 — System Instructions

- [x] Correctly scoped as N/A — **PASS**

### Checklist § 4 — Finish Reasons

- [x] Top-level `gen_ai.response.finish_reasons` as `string[]` — **PASS** (§2.3)
- [x] Omit when None, don't fabricate `"stop"` — **PASS** (§2.3)
- [x] Per-message `finish_reason` always present as string — **PASS** (§5.1 uses `""` fallback)
- [x] **P1 — RESOLVED: `gen_ai.response.finish_reasons` not gated by `should_send_prompts()`**. Added CRITICAL note to §5.3 step 3, §5.4 step 4, §5.8 matrix. Added `test_not_gated_by_should_send_prompts` to §4.1.2.
- [x] **P2 — RESOLVED: `_FINISH_REASON_MAP` expanded**. Added Cohere (`COMPLETE`, `MAX_TOKENS`, `ERROR`, `ERROR_TOXIC`) + Anthropic (`end_turn`, `stop_sequence`, `tool_use`, `max_tokens`) to §5.1. Pass-through behavior documented in `_FINISH_REASON_MAP` comment.
- [x] **P2 — RESOLVED: Cohere finish reason extraction**. Added multi-provider `extract_finish_reasons()` to §5.2 handling OpenAI/Cohere/Anthropic formats.

### Checklist § 5 — Operation Name & Provider Name

- [x] `gen_ai.operation.name` values correct (`"chat"`, `"text_completion"`, `"embeddings"`) — **PASS**
- [x] **P1 — RESOLVED: Provider name detection in dispatcher path**. Added `_PROVIDER_MAP` with 11 OTel well-known values to §5.2. Added `detect_provider_name()` supporting instance and string input. Added provider extraction in `new_span` (§5.6 step 2). Added tests `test_from_model_dict_class_name` and `test_from_span_handler_instance` to §4.1.7.
- [x] **P2 — RESOLVED: Custom LLM provider name**. `detect_provider_name()` in §5.2 uses lowercase class name as fallback. Documented in §5.4 step 1.

### Checklist § 6 — Attribute Renames

- [x] All major renames identified — **PASS** (§1.3.4)
- [x] Upstream constants used — **PASS**
- [x] Indexed attributes replaced with JSON — **PASS**

### Checklist § 7 — Semconv-AI Backward-Compatible Evolution

- [x] No breaking changes needed — **PASS**
- [x] Dependency compatibility gate planned — **PASS** (§7.3)
- [ ] **P3 — Note**: Plan should verify that `GenAIAttributes.GEN_AI_PROVIDER_NAME` exists in the current `opentelemetry-semantic-conventions>=0.59b0` range. If it was added in a later version, the lower bound may need adjustment.

### Checklist § 8 — Code Path Completeness

- [x] **P1 — RESOLVED: Code paths explicitly enumerated**. Added §5.8 Code Path Completeness Matrix with 11 rows covering all paths including error paths and delegated OpenAI spans.

- [x] **P2 — RESOLVED: Streaming finish_reason**. Added §5.6 step 3 explaining LLMChatEndEvent carries accumulated response with finish_reason even in streaming mode.

- [x] **P2 — RESOLVED: Error/exception paths**. Added §5.6 step 4 documenting `dont_throw` partial attribute behavior. Added §5.8 row 10.

- [x] **P2 — RESOLVED: `LLMPredictEndEvent` path**. Expanded §5.3 step 5 with operation.name (`"text_completion"`), provider.name, finish_reasons, `""` fallback for per-message finish_reason.

### Checklist § 9 — Dependencies & Versioning

- [x] Dependency ranges addressed — **PASS**

### Checklist § 10 — Code Quality

- [x] Async/sync identical via shared helpers — **PASS**
- [x] No hardcoded strings — **PASS** (plan uses upstream constants)

### Checklist § 11 — Test Coverage

- [x] Unit tests for `_set_input_messages` / `_set_output_messages` — **PASS** (§4.1.5)
- [x] **P2 — RESOLVED: Multimodal content mapping tests**. Added 4 multimodal tests to §4.1.1.
- [x] **P2 — RESOLVED: Tool call round-trip tests**. Added `test_tool_call_response_round_trip` to §4.1.1.
- [x] Tests for `gen_ai.response.finish_reasons` as top-level span attribute — **PASS** (§4.1.2)
- [x] Tests for None content, None finish_reason — **PASS** (§4.1.1, §4.1.2)
- [x] **P2 — RESOLVED: Streaming format equivalence test**. Added `test_streaming_produces_same_format_as_non_streaming` to §4.1.5.
- [x] **P1 — RESOLVED: Finish_reasons gating test**. Added `test_not_gated_by_should_send_prompts` to §4.1.2.

### Checklist § 12 — Gateway Impact

- [x] Gateway impact assessed — **PASS** (§8)
- [x] SDK follows OTel spec regardless of gateway — **PASS**

---

## Appendix D: Finish Reasons Semconv Review (openllmetry-semconv-review-finish-reasons checklist)

### Mapper Function

- [x] **P2 — RESOLVED: Every provider finish_reason mapped**. Added Cohere (`COMPLETE`, `MAX_TOKENS`, `ERROR`, `ERROR_TOXIC`) + Anthropic (`end_turn`, `stop_sequence`, `tool_use`, `max_tokens`) to §5.1. Pass-through documented in `_FINISH_REASON_MAP` comment.
- [x] `None` input handled — `map_finish_reason(None)` returns `None` — **PASS** for top-level
- [x] **P3 — RESOLVED: `UNSPECIFIED` / unknown enum values**. Documented in `_FINISH_REASON_MAP` comment and `test_unknown_passes_through` comment.
- [x] **P2 — RESOLVED: Return type clarification**. Added Contract docstring to `map_finish_reason` in §5.1 clarifying dual-purpose return type.

### Output Messages JSON (`gen_ai.output.messages`)

- [x] Every output message dict includes `"finish_reason"` key — **PASS** (§5.1 `build_output_message`)
- [x] `finish_reason` value is always a string — **PASS** (`""` fallback)
- [x] **P2 — RESOLVED: All code paths covered**. Streaming path addressed in §5.6 step 3 (LLMChatEndEvent carries accumulated response). §5.8 matrix covers all 11 paths.
- [x] **P2 — RESOLVED: `set_llm_predict_response` path**. Expanded §5.3 step 5 with `build_completion_output_message`, `""` fallback for finish_reason.

### Top-Level Span Attribute (`gen_ai.response.finish_reasons`)

- [x] Set as `string[]` array — **PASS** (§2.1)
- [x] **P1 — RESOLVED: Not gated by `should_send_prompts()`**. Added CRITICAL note to §5.3 step 3, §5.4 step 4, §5.8 key rule. Test added to §4.1.2.
- [x] Unknown values filtered out — **PASS** (only set when available)
- [x] Not fabricated — **PASS** (§2.3: "Never fabricate `stop`")
- [x] **P3 — RESOLVED: Streaming deduplication**. Addressed in §5.6 step 3 (LlamaIndex fires single LLMChatEndEvent at stream end, no accumulation/dedup needed).

### Code Path Completeness for Finish Reasons

| Path | Top-level `gen_ai.response.finish_reasons` | Per-message `finish_reason` | Plan Coverage |
|------|--------------------------------------------|-----------------------------|---------------|
| Dispatcher: LLM chat non-streaming | Via `set_llm_chat_response_model_attributes` | Via `set_llm_chat_response` → `build_output_message` | ✅ Covered (§5.3) |
| Dispatcher: LLM chat streaming | Via `set_llm_chat_response_model_attributes` (on `LLMChatEndEvent`) | Via `set_llm_chat_response` (on `LLMChatEndEvent`) | ✅ Covered (§5.6 step 3) |
| Dispatcher: LLM predict | Via `set_llm_predict_response` if available | `""` fallback via `build_completion_output_message` | ✅ Covered (§5.3 step 5) |
| CustomLLM: chat sync | Via `_handle_response` + `extract_finish_reasons` | Via `build_output_message` | ✅ Covered (§5.4 step 4) |
| CustomLLM: chat async | Same as sync | Same as sync | ✅ Covered (shared helpers) |
| CustomLLM: complete sync | Via `_handle_response` + `extract_finish_reasons` | Via `build_completion_output_message` | ✅ Covered (§5.4 step 4) |
| CustomLLM: complete async | Same as sync | Same as sync | ✅ Covered (shared helpers) |
| Error paths | Partial (set before error) | Partial (set before error) | ✅ Documented (§5.6 step 4, §5.8 row 10) |

---

## Appendix E: Summary of Review Findings — Resolution Status

### P1 — Must Fix (3 findings) — ✅ ALL ADDRESSED

1. **`gen_ai.response.finish_reasons` must NOT be gated by `should_send_prompts()`**
   - ✅ Added to §5.3 step 3 (explicit CRITICAL note), §5.4 step 4, §5.8 matrix (bold "ungated")
   - ✅ Added `test_not_gated_by_should_send_prompts` to §4.1.2 `TestFinishReasonTopLevel`
2. **Provider name detection in dispatcher path not designed**
   - ✅ Added `_PROVIDER_MAP` with 11 OTel well-known values to §5.2 `_response_utils.py`
   - ✅ Added `detect_provider_name()` with instance + string support and lowercase fallback
   - ✅ Added provider extraction from `class_name` in `new_span` to §5.6 step 2
   - ✅ Added `test_from_model_dict_class_name` and `test_from_span_handler_instance` to §4.1.7
3. **Code paths not explicitly enumerated**
   - ✅ Added §5.8 Code Path Completeness Matrix with 11 rows covering all paths
   - ✅ Key rule documented: finish_reasons never gated by `should_send_prompts()`

### P2 — Should Fix (10 findings) — ✅ ALL ADDRESSED

1. **Multimodal content** — ✅ Added `_content_to_parts()` to §5.1, handles str/list/image blocks → BlobPart/UriPart
2. **Tool call parts** — ✅ Added `_extract_tool_calls_from_message()` to §5.1, extracts from `additional_kwargs`
3. **Incomplete `_FINISH_REASON_MAP`** — ✅ Added Cohere (`COMPLETE`, `MAX_TOKENS`, `ERROR`, `ERROR_TOXIC`) + Anthropic (`end_turn`, `stop_sequence`, `tool_use`, `max_tokens`) mappings to §5.1
4. **Cohere finish reason extraction** — ✅ Added multi-provider `extract_finish_reasons()` to §5.2, handles OpenAI/Cohere/Anthropic formats
5. **Streaming finish_reason accumulation** — ✅ Added §5.6 step 3 explaining LLMChatEndEvent carries accumulated response
6. **`LLMPredictEndEvent` missing attrs** — ✅ Expanded §5.3 step 5 with operation.name, provider.name, finish_reasons, `""` fallback
7. **Error/exception paths** — ✅ Added §5.6 step 4, §5.8 row 10, documented `dont_throw` partial attribute behavior
8. **Missing multimodal/tool call tests** — ✅ Added 5 new tests to §4.1.1 (`test_multimodal_content_list`, `test_image_url_mapped_to_uri_part`, `test_image_base64_mapped_to_blob_part`, `test_mixed_text_and_image_content`, `test_tool_call_response_round_trip`)
9. **Missing streaming format equivalence test** — ✅ Added `test_streaming_produces_same_format_as_non_streaming` to §4.1.5
10. **Missing finish_reasons gating test** — ✅ Added `test_not_gated_by_should_send_prompts` to §4.1.2

### P3 — Nice to Have (3 findings) — ✅ ALL ADDRESSED

1. **`arguments` parsing** — ✅ Added `_parse_arguments()` helper with `json.loads` + fallback to §5.1
2. **Unknown enum values documented** — ✅ Added docstring to `_FINISH_REASON_MAP` and `map_finish_reason`, `test_unknown_passes_through` comment clarified
3. **Streaming deduplication** — ✅ Addressed in §5.6 step 3 (LlamaIndex fires single LLMChatEndEvent at stream end, no accumulation needed)
