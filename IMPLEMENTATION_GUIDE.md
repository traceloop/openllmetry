# 🚀 OpenTelemetry AI Provider Instrumentation Update
## Event-Based Prompt Tracking Implementation Guide

## 📋 Overview

This document outlines the implementation plan for adding event-based prompt tracking across all AI provider instrumentations while maintaining backward compatibility with the current attribute-based approach.

## 🎯 Requirements

- Support both legacy attribute-based and new event-based approaches
- Add `use_legacy_attributes` config parameter (default: true) 
- Implement across 17 AI provider packages
- Follow OpenAI's implementation as reference

## 🔑 Key Implementation Details

### Event-Based vs Attribute-Based Approach

1. **Legacy (Attribute-Based)**:
```python
span.set_attribute("gen_ai.prompt.0.content", prompt_text)
span.set_attribute("gen_ai.completion.0.content", completion_text)
```

2. **New (Event-Based)**:
```python
event_logger.emit(Event(
    name="gen_ai.prompt",
    attributes={"gen_ai.system": system},
    body={"role": "user", "content": prompt_text}
))
```

### Configuration Implementation

1. **SDK Level**:
```python
# Initialize SDK with global config
opentelemetry.configure(use_legacy_attributes=False)
```

2. **Per-Provider Level**:
```python
class Config:
    use_legacy_attributes = True  # Default value

class ProviderInstrumentor(BaseInstrumentor):
    def _instrument(self, **kwargs):
        Config.use_legacy_attributes = kwargs.get('use_legacy_attributes', True)
```

### Event Handling Requirements

1. **Event Logger Usage**:
- Use `EventLogger` instead of direct `span.add_event`
- Maintain trace context for events
- Handle streaming responses appropriately

2. **Event Structure**:
```python
Event(
    name="gen_ai.prompt",
    attributes={...},  # Semantic conventions
    body={...},        # Provider-specific data
    trace_id=span_ctx.trace_id,
    span_id=span_ctx.span_id,
    trace_flags=span_ctx.trace_flags
)
```

## 🏗️ Implementation Plan

### Phase 1: Core Infrastructure ✅

1. Create shared event utilities:
```python
def create_base_event(name: str, system: str) -> Event:
    """Creates a base event with proper trace context"""
    return Event(
        name=name,
        attributes={"gen_ai.system": system},
        trace_id=current_span_context().trace_id,
        span_id=current_span_context().span_id,
        trace_flags=current_span_context().trace_flags
    )
```

### Phase 2: Provider Implementation Schedule

#### Batch 1 (Core Providers)
- ✅ OpenAI (Complete - Reference Implementation)
- ✅ AlephAlpha (Complete)
  - Added event-based tracking
  - Implemented streaming support
  - Added error handling
  - Full test coverage including:
    - Legacy mode
    - Event mode
    - Streaming
    - Error cases
- ⏳ Anthropic (Next)
- ⏳ Google Generative AI
- ⏳ Azure OpenAI

Key Implementation Steps (Based on AlephAlpha):
1. Add `use_legacy_attributes` to provider's Config class
2. Create provider-specific event utilities
3. Update wrapper to support both modes
4. Add tests for both legacy and event modes

#### Batch 2
- Cohere
- Mistral AI  
- Ollama
- LlamaIndex

#### Batch 3
- Bedrock
- SageMaker
- VertexAI

#### Batch 4
- AlephAlpha
- Groq
- Langchain
- Replicate
- Together
- WatsonX

### Phase 3: Per-Provider Implementation Steps

1. Update Provider Configuration:

python
```
class ProviderInstrumentor(BaseInstrumentor):
def init(self):
super().init()
self.use_legacy_attributes = True
def instrument(self, kwargs):
self.use_legacy_attributes = kwargs.get('use_legacy_attributes', True)
```


2. Update Wrapper Implementation:

python
```
def wrap_completion(tracer, event_logger, capture_content):
def wrapper(wrapped, instance, args, kwargs):
with tracer.start_as_current_span(...) as span:
# Event-based approach
if not instance.use_legacy_attributes:
event_logger.emit(
create_prompt_event(...)
)
# Legacy approach
if instance.use_legacy_attributes:
span.set_attribute(...)
```


## 🧪 Testing Requirements

Each provider must test:

1. **Configuration**:
```python
def test_config_propagation():
    # Test SDK-level config
    sdk.configure(use_legacy_attributes=False)
    assert Config.use_legacy_attributes == False
    
    # Test provider-level override
    instrumentor = ProviderInstrumentor()
    instrumentor._instrument(use_legacy_attributes=True)
    assert Config.use_legacy_attributes == True
```

2. **Event Handling**:
```python
def test_event_based_tracking():
    # Configure for event-based
    instrumentor._instrument(use_legacy_attributes=False)
    
    # Verify events
    events = span.events
    assert events[0].name == "gen_ai.prompt"
    assert events[0].trace_id == span.context.trace_id
```

## 📝 Documentation Updates

For each provider:
1. Update README with new configuration options
2. Add migration guide section
3. Update examples to show both modes
4. Document any provider-specific considerations

## 🔄 Implementation Process

1. Create feature branch
2. Implement core utilities
3. Update providers one batch at a time
4. Add tests for each provider
5. Update documentation
6. Submit PR for review

## ✅ Definition of Done

- All 17 providers updated
- Comprehensive test coverage
- Documentation updated
- Backward compatibility maintained
- PR review completed
- CI/CD passes

## 📊 Progress Tracking

- [x] Core Infrastructure
- [ ] Batch 1 Providers
  - [x] OpenAI
  - [x] AlephAlpha
  - [ ] Anthropic (Next)
  - [ ] Google Generative AI
  - [ ] Azure OpenAI
- [ ] Batch 2 Providers
- [ ] Batch 3 Providers
- [ ] Batch 4 Providers
- [ ] Documentation
- [ ] Testing

## 🤝 Contributing

Please refer to CONTRIBUTING.md for detailed guidelines on:
- Code style
- Testing requirements
- PR process
- Review criteria

## ❓ Questions?

For questions or clarifications:
1. Open an issue
2. Tag with 'question' label
3. Reference the provider and implementation phase

## 📚 References

- [OpenAI Implementation](https://github.com/open-telemetry/opentelemetry-python-contrib/blob/main/instrumentation/opentelemetry-instrumentation-openai)
- [Semantic Conventions](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/ai/ai-spans.md)
- [Event API Documentation](https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/logs/event-api.md)

## 📄 License

Apache 2.0 - See LICENSE file for details



 <<<<<<<<OpenAI instrumentation>>>>>>>>
# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Optional

from openai import Stream

from opentelemetry._events import Event, EventLogger
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace import Span, SpanKind, Tracer

from .utils import (
    choice_to_event,
    get_llm_request_attributes,
    handle_span_exception,
    is_streaming,
    message_to_event,
    set_span_attribute,
)


def chat_completions_create(
    tracer: Tracer, event_logger: EventLogger, capture_content: bool
):
    """Wrap the `create` method of the `ChatCompletion` class to trace it."""

    def traced_method(wrapped, instance, args, kwargs):
        span_attributes = {**get_llm_request_attributes(kwargs, instance)}

        span_name = f"{span_attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]} {span_attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]}"
        with tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.CLIENT,
            attributes=span_attributes,
            end_on_exit=False,
        ) as span:
            if span.is_recording():
                for message in kwargs.get("messages", []):
                    event_logger.emit(
                        message_to_event(message, capture_content)
                    )

            try:
                result = wrapped(*args, **kwargs)
                if is_streaming(kwargs):
                    return StreamWrapper(
                        result, span, event_logger, capture_content
                    )

                if span.is_recording():
                    _set_response_attributes(
                        span, result, event_logger, capture_content
                    )
                span.end()
                return result

            except Exception as error:
                handle_span_exception(span, error)
                raise

    return traced_method


def async_chat_completions_create(
    tracer: Tracer, event_logger: EventLogger, capture_content: bool
):
    """Wrap the `create` method of the `AsyncChatCompletion` class to trace it."""

    async def traced_method(wrapped, instance, args, kwargs):
        span_attributes = {**get_llm_request_attributes(kwargs, instance)}

        span_name = f"{span_attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]} {span_attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]}"
        with tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.CLIENT,
            attributes=span_attributes,
            end_on_exit=False,
        ) as span:
            if span.is_recording():
                for message in kwargs.get("messages", []):
                    event_logger.emit(
                        message_to_event(message, capture_content)
                    )

            try:
                result = await wrapped(*args, **kwargs)
                if is_streaming(kwargs):
                    return StreamWrapper(
                        result, span, event_logger, capture_content
                    )

                if span.is_recording():
                    _set_response_attributes(
                        span, result, event_logger, capture_content
                    )
                span.end()
                return result

            except Exception as error:
                handle_span_exception(span, error)
                raise

    return traced_method


def _set_response_attributes(
    span, result, event_logger: EventLogger, capture_content: bool
):
    set_span_attribute(
        span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, result.model
    )

    if getattr(result, "choices", None):
        choices = result.choices
        for choice in choices:
            event_logger.emit(choice_to_event(choice, capture_content))

        finish_reasons = []
        for choice in choices:
            finish_reasons.append(choice.finish_reason or "error")

        set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS,
            finish_reasons,
        )

    if getattr(result, "id", None):
        set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, result.id)

    if getattr(result, "service_tier", None):
        set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OPENAI_REQUEST_SERVICE_TIER,
            result.service_tier,
        )

    # Get the usage
    if getattr(result, "usage", None):
        set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
            result.usage.prompt_tokens,
        )
        set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
            result.usage.completion_tokens,
        )


class ToolCallBuffer:
    def __init__(self, index, tool_call_id, function_name):
        self.index = index
        self.function_name = function_name
        self.tool_call_id = tool_call_id
        self.arguments = []

    def append_arguments(self, arguments):
        self.arguments.append(arguments)


class ChoiceBuffer:
    def __init__(self, index):
        self.index = index
        self.finish_reason = None
        self.text_content = []
        self.tool_calls_buffers = []

    def append_text_content(self, content):
        self.text_content.append(content)

    def append_tool_call(self, tool_call):
        idx = tool_call.index
        # make sure we have enough tool call buffers
        for _ in range(len(self.tool_calls_buffers), idx + 1):
            self.tool_calls_buffers.append(None)

        if not self.tool_calls_buffers[idx]:
            self.tool_calls_buffers[idx] = ToolCallBuffer(
                idx, tool_call.id, tool_call.function.name
            )
        self.tool_calls_buffers[idx].append_arguments(
            tool_call.function.arguments
        )


class StreamWrapper:
    span: Span
    response_id: Optional[str] = None
    response_model: Optional[str] = None
    service_tier: Optional[str] = None
    finish_reasons: list = []
    prompt_tokens: Optional[int] = 0
    completion_tokens: Optional[int] = 0

    def __init__(
        self,
        stream: Stream,
        span: Span,
        event_logger: EventLogger,
        capture_content: bool,
    ):
        self.stream = stream
        self.span = span
        self.choice_buffers = []
        self._span_started = False
        self.capture_content = capture_content

        self.event_logger = event_logger
        self.setup()

    def setup(self):
        if not self._span_started:
            self._span_started = True

    def cleanup(self):
        if self._span_started:
            if self.response_model:
                set_span_attribute(
                    self.span,
                    GenAIAttributes.GEN_AI_RESPONSE_MODEL,
                    self.response_model,
                )

            if self.response_id:
                set_span_attribute(
                    self.span,
                    GenAIAttributes.GEN_AI_RESPONSE_ID,
                    self.response_id,
                )

            set_span_attribute(
                self.span,
                GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
                self.prompt_tokens,
            )
            set_span_attribute(
                self.span,
                GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                self.completion_tokens,
            )

            set_span_attribute(
                self.span,
                GenAIAttributes.GEN_AI_OPENAI_RESPONSE_SERVICE_TIER,
                self.service_tier,
            )

            set_span_attribute(
                self.span,
                GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS,
                self.finish_reasons,
            )

            for idx, choice in enumerate(self.choice_buffers):
                message = {"role": "assistant"}
                if self.capture_content and choice.text_content:
                    message["content"] = "".join(choice.text_content)
                if choice.tool_calls_buffers:
                    tool_calls = []
                    for tool_call in choice.tool_calls_buffers:
                        function = {"name": tool_call.function_name}
                        if self.capture_content:
                            function["arguments"] = "".join(
                                tool_call.arguments
                            )
                        tool_call_dict = {
                            "id": tool_call.tool_call_id,
                            "type": "function",
                            "function": function,
                        }
                        tool_calls.append(tool_call_dict)
                    message["tool_calls"] = tool_calls

                body = {
                    "index": idx,
                    "finish_reason": choice.finish_reason or "error",
                    "message": message,
                }

                event_attributes = {
                    GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.OPENAI.value
                }

                # this span is not current, so we need to manually set the context on event
                span_ctx = self.span.get_span_context()
                self.event_logger.emit(
                    Event(
                        name="gen_ai.choice",
                        attributes=event_attributes,
                        body=body,
                        trace_id=span_ctx.trace_id,
                        span_id=span_ctx.span_id,
                        trace_flags=span_ctx.trace_flags,
                    )
                )

            self.span.end()
            self._span_started = False

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                handle_span_exception(self.span, exc_val)
        finally:
            self.cleanup()
        return False  # Propagate the exception

    async def __aenter__(self):
        self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                handle_span_exception(self.span, exc_val)
        finally:
            self.cleanup()
        return False  # Propagate the exception

    def close(self):
        self.stream.close()
        self.cleanup()

    def __iter__(self):
        return self

    def __aiter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self.stream)
            self.process_chunk(chunk)
            return chunk
        except StopIteration:
            self.cleanup()
            raise
        except Exception as error:
            handle_span_exception(self.span, error)
            self.cleanup()
            raise

    async def __anext__(self):
        try:
            chunk = await self.stream.__anext__()
            self.process_chunk(chunk)
            return chunk
        except StopAsyncIteration:
            self.cleanup()
            raise
        except Exception as error:
            handle_span_exception(self.span, error)
            self.cleanup()
            raise

    def set_response_model(self, chunk):
        if self.response_model:
            return

        if getattr(chunk, "model", None):
            self.response_model = chunk.model

    def set_response_id(self, chunk):
        if self.response_id:
            return

        if getattr(chunk, "id", None):
            self.response_id = chunk.id

    def set_response_service_tier(self, chunk):
        if self.service_tier:
            return

        if getattr(chunk, "service_tier", None):
            self.service_tier = chunk.service_tier

    def build_streaming_response(self, chunk):
        if getattr(chunk, "choices", None) is None:
            return

        choices = chunk.choices
        for choice in choices:
            if not choice.delta:
                continue

            # make sure we have enough choice buffers
            for idx in range(len(self.choice_buffers), choice.index + 1):
                self.choice_buffers.append(ChoiceBuffer(idx))

            if choice.finish_reason:
                self.choice_buffers[
                    choice.index
                ].finish_reason = choice.finish_reason

            if choice.delta.content is not None:
                self.choice_buffers[choice.index].append_text_content(
                    choice.delta.content
                )

            if choice.delta.tool_calls is not None:
                for tool_call in choice.delta.tool_calls:
                    self.choice_buffers[choice.index].append_tool_call(
                        tool_call
                    )

    def set_usage(self, chunk):
        if getattr(chunk, "usage", None):
            self.completion_tokens = chunk.usage.completion_tokens
            self.prompt_tokens = chunk.usage.prompt_tokens

    def process_chunk(self, chunk):
        self.set_response_id(chunk)
        self.set_response_model(chunk)
        self.set_response_service_tier(chunk)
        self.build_streaming_response(chunk)
        self.set_usage(chunk)