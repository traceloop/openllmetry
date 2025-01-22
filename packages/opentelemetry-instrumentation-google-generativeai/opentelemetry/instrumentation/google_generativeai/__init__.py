"""OpenTelemetry Google Generative AI API instrumentation"""

import logging
import os
import types
from typing import Collection
from opentelemetry.instrumentation.google_generativeai.config import Config
from opentelemetry.instrumentation.google_generativeai.utils import dont_throw
from opentelemetry.instrumentation.google_generativeai.events import (
    create_prompt_event,
    create_completion_event,
    create_tool_call_event,
    create_function_call_event,
)
from wrapt import wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, SpanKind
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap

from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
    LLMRequestTypeValues,
)
from opentelemetry.instrumentation.google_generativeai.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("google-generativeai >= 0.5.0",)

WRAPPED_METHODS = [
    {
        "package": "google.generativeai.generative_models",
        "object": "GenerativeModel",
        "method": "generate_content",
        "span_name": "gemini.generate_content",
    },
    {
        "package": "google.generativeai.generative_models",
        "object": "GenerativeModel",
        "method": "generate_content_async",
        "span_name": "gemini.generate_content_async",
    },
    {
        "package": "google.generativeai.generative_models",
        "object": "ChatSession",
        "method": "send_message",
        "span_name": "gemini.send_message",
    },
]


def should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def is_streaming_response(response):
    return isinstance(response, types.GeneratorType)


def is_async_streaming_response(response):
    return isinstance(response, types.AsyncGeneratorType)


def _set_span_attribute(span, name, value):
    if value is not None and value != "":
        span.set_attribute(name, value)
    return


def _set_input_attributes(span, args, kwargs, llm_model, event_logger=None, use_legacy_attributes=True):
    if should_send_prompts() and args is not None and len(args) > 0:
        prompt = ""
        for arg in args:
            if isinstance(arg, str):
                prompt = f"{prompt}{arg}\n"
            elif isinstance(arg, list):
                for subarg in arg:
                    prompt = f"{prompt}{subarg}\n"

        if use_legacy_attributes:
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.0.user",
                prompt,
            )
        if event_logger:
            event_logger.emit_event(create_prompt_event(
                content=prompt,
                role="user",
                content_type="text",
                model=llm_model,
            ))

    if use_legacy_attributes:
        _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, llm_model)
        _set_span_attribute(
            span, f"{SpanAttributes.LLM_PROMPTS}.0.user", kwargs.get("prompt")
        )
        _set_span_attribute(
            span, SpanAttributes.LLM_REQUEST_TEMPERATURE, kwargs.get("temperature")
        )
        _set_span_attribute(
            span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_output_tokens")
        )
        _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TOP_P, kwargs.get("top_p"))
        _set_span_attribute(span, SpanAttributes.LLM_TOP_K, kwargs.get("top_k"))
        _set_span_attribute(
            span, SpanAttributes.LLM_PRESENCE_PENALTY, kwargs.get("presence_penalty")
        )
        _set_span_attribute(
            span, SpanAttributes.LLM_FREQUENCY_PENALTY, kwargs.get("frequency_penalty")
        )

    if event_logger and kwargs.get("prompt"):
        event_logger.emit_event(create_prompt_event(
            content=kwargs.get("prompt"),
            role="user",
            content_type="text",
            model=llm_model,
        ))

    return


@dont_throw
def _set_response_attributes(span, response, llm_model, event_logger=None, use_legacy_attributes=True):
    if use_legacy_attributes:
        _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, llm_model)

    if hasattr(response, "usage_metadata"):
        if use_legacy_attributes:
            _set_span_attribute(
                span,
                SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
                response.usage_metadata.total_token_count,
            )
            _set_span_attribute(
                span,
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                response.usage_metadata.candidates_token_count,
            )
            _set_span_attribute(
                span,
                SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
                response.usage_metadata.prompt_token_count,
            )

        # Extract safety attributes if present
        safety_attributes = {}
        if hasattr(response, "safety_ratings"):
            for rating in response.safety_ratings:
                safety_attributes[rating.category.lower()] = rating.probability

        if isinstance(response.text, list):
            for index, item in enumerate(response):
                if use_legacy_attributes:
                    prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
                    _set_span_attribute(span, f"{prefix}.content", item.text)
                if event_logger:
                    event_logger.emit_event(create_completion_event(
                        completion=item.text,
                        model=llm_model,
                        completion_tokens=response.usage_metadata.candidates_token_count,
                        role="assistant",
                        content_type="text",
                        safety_attributes=safety_attributes,
                    ))
        elif isinstance(response.text, str):
            if use_legacy_attributes:
                _set_span_attribute(
                    span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", response.text
                )
            if event_logger:
                event_logger.emit_event(create_completion_event(
                    completion=response.text,
                    model=llm_model,
                    completion_tokens=response.usage_metadata.candidates_token_count,
                    role="assistant",
                    content_type="text",
                    safety_attributes=safety_attributes,
                ))

        # Handle tool calls if present
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                if event_logger:
                    event_logger.emit_event(create_tool_call_event(
                        tool_name=tool_call.name,
                        tool_input=tool_call.args,
                        tool_output=tool_call.response if hasattr(tool_call, "response") else None,
                        model=llm_model,
                    ))

        # Handle function calls if present
        if hasattr(response, "function_call"):
            if event_logger:
                event_logger.emit_event(create_function_call_event(
                    function_name=response.function_call.name,
                    function_args=response.function_call.args,
                    function_output=response.function_call.response if hasattr(response.function_call, "response") else None,
                    model=llm_model,
                ))
    else:
        if isinstance(response, list):
            for index, item in enumerate(response):
                if use_legacy_attributes:
                    prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
                    _set_span_attribute(span, f"{prefix}.content", item)
                if event_logger:
                    event_logger.emit_event(create_completion_event(
                        completion=item,
                        model=llm_model,
                        role="assistant",
                        content_type="text",
                    ))
        elif isinstance(response, str):
            if use_legacy_attributes:
                _set_span_attribute(
                    span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", response
                )
            if event_logger:
                event_logger.emit_event(create_completion_event(
                    completion=response,
                    model=llm_model,
                    role="assistant",
                    content_type="text",
                ))

    return


def _build_from_streaming_response(span, response, llm_model, event_logger=None, use_legacy_attributes=True):
    """Build a response from a streaming response."""
    completion_buffer = []
    
    def process_chunk(chunk):
        if not chunk:
            return
            
        if hasattr(chunk, "text"):
            completion_buffer.append(chunk.text)
            if use_legacy_attributes:
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
                    "".join(completion_buffer)
                )
            
            if event_logger:
                event_logger.emit(create_completion_event(
                    completion=chunk.text,
                    model=llm_model,
                    role="assistant",
                    content_type="text",
                    finish_reason=getattr(chunk, "finish_reason", None),
                    safety_attributes=getattr(chunk, "safety_attributes", None)
                ))
        
        # Handle candidates if present
        if hasattr(chunk, "candidates"):
            for i, candidate in enumerate(chunk.candidates):
                if hasattr(candidate, "content") and hasattr(candidate.content, "text"):
                    if use_legacy_attributes:
                        _set_span_attribute(
                            span,
                            f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content",
                            candidate.content.text
                        )
                    
                    if event_logger:
                        event_logger.emit(create_completion_event(
                            completion=candidate.content.text,
                            model=llm_model,
                            role="assistant",
                            content_type="text",
                            finish_reason=getattr(candidate, "finish_reason", None),
                            safety_attributes=getattr(candidate, "safety_attributes", None)
                        ))

    for chunk in response:
        process_chunk(chunk)
        yield chunk

    span.end()


async def _abuild_from_streaming_response(span, response, llm_model, event_logger=None, use_legacy_attributes=True):
    """Build a response from an async streaming response."""
    completion_buffer = []
    
    def process_chunk(chunk):
        if not chunk:
            return
            
        if hasattr(chunk, "text"):
            completion_buffer.append(chunk.text)
            if use_legacy_attributes:
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
                    "".join(completion_buffer)
                )
            
            if event_logger:
                event_logger.emit(create_completion_event(
                    completion=chunk.text,
                    model=llm_model,
                    role="assistant",
                    content_type="text",
                    finish_reason=getattr(chunk, "finish_reason", None),
                    safety_attributes=getattr(chunk, "safety_attributes", None)
                ))
        
        # Handle candidates if present
        if hasattr(chunk, "candidates"):
            for i, candidate in enumerate(chunk.candidates):
                if hasattr(candidate, "content") and hasattr(candidate.content, "text"):
                    if use_legacy_attributes:
                        _set_span_attribute(
                            span,
                            f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content",
                            candidate.content.text
                        )
                    
                    if event_logger:
                        event_logger.emit(create_completion_event(
                            completion=candidate.content.text,
                            model=llm_model,
                            role="assistant",
                            content_type="text",
                            finish_reason=getattr(candidate, "finish_reason", None),
                            safety_attributes=getattr(candidate, "safety_attributes", None)
                        ))

    async for chunk in response:
        process_chunk(chunk)
        yield chunk

    span.end()


@dont_throw
def _handle_request(span, args, kwargs, llm_model, event_logger=None, use_legacy_attributes=True):
    """Handle the request attributes and events."""
    if not span.is_recording():
        return

    # Set basic request attributes
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.COMPLETION)
    
    # Handle content from args and kwargs
    content = None
    if args and len(args) > 0:
        content = args[0]
    elif "contents" in kwargs:
        content = kwargs["contents"]
    elif "content" in kwargs:
        content = kwargs["content"]

    if content:
        if use_legacy_attributes:
            if isinstance(content, str):
                _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.user", content)
            elif isinstance(content, (list, tuple)):
                for i, msg in enumerate(content):
                    if isinstance(msg, str):
                        _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.{i}.user", msg)
                    elif isinstance(msg, dict) and "text" in msg:
                        _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.{i}.user", msg["text"])

        if event_logger:
            if isinstance(content, str):
                event_logger.emit(create_prompt_event(
                    content=content,
                    role="user",
                    content_type="text",
                    model=llm_model
                ))
            elif isinstance(content, (list, tuple)):
                for msg in content:
                    if isinstance(msg, str):
                        event_logger.emit(create_prompt_event(
                            content=msg,
                            role="user",
                            content_type="text",
                            model=llm_model
                        ))
                    elif isinstance(msg, dict):
                        event_logger.emit(create_prompt_event(
                            content=msg,
                            role=msg.get("role", "user"),
                            content_type="text" if "text" in msg else "image" if "image" in msg else None,
                            model=llm_model
                        ))

@dont_throw
def _handle_response(span, response, llm_model, event_logger=None, use_legacy_attributes=True):
    """Handle the response attributes and events."""
    if not span.is_recording() or not response:
        return

    if hasattr(response, "text"):
        if use_legacy_attributes:
            _set_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", response.text)
        
        if event_logger:
            event_logger.emit(create_completion_event(
                completion=response.text,
                model=llm_model,
                role="assistant",
                content_type="text",
                finish_reason=getattr(response, "finish_reason", None),
                safety_attributes=getattr(response, "safety_attributes", None)
            ))

    # Handle candidates/choices if present
    if hasattr(response, "candidates"):
        for i, candidate in enumerate(response.candidates):
            if hasattr(candidate, "content") and hasattr(candidate.content, "text"):
                if use_legacy_attributes:
                    _set_span_attribute(
                        span,
                        f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content",
                        candidate.content.text
                    )
                
                if event_logger:
                    event_logger.emit(create_completion_event(
                        completion=candidate.content.text,
                        model=llm_model,
                        role="assistant",
                        content_type="text",
                        finish_reason=getattr(candidate, "finish_reason", None),
                        safety_attributes=getattr(candidate, "safety_attributes", None)
                    ))

    # Handle token usage if available
    if hasattr(response, "usage"):
        _set_span_attribute(
            span,
            SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
            getattr(response.usage, "prompt_tokens", None)
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
            getattr(response.usage, "completion_tokens", None)
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
            getattr(response.usage, "total_tokens", None)
        )


def _with_tracer_wrapper(func):
    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                return wrapped(*args, **kwargs)

            if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
                return wrapped(*args, **kwargs)

            return func(tracer, to_wrap, wrapped, instance, args, kwargs)
        return wrapper
    return _with_tracer


@_with_tracer_wrapper
async def _awrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    # Get model name with proper fallbacks
    llm_model = "unknown"
    if hasattr(instance, "_model_id"):
        llm_model = instance._model_id
    if hasattr(instance, "_model_name"):
        llm_model = instance._model_name.replace("publishers/google/models/", "")

    name = to_wrap.get("span_name")
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Gemini",
            SpanAttributes.LLM_REQUEST_MODEL: llm_model,
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
    )

    # Get event logger and config from kwargs if provided, otherwise from instance
    event_logger = kwargs.pop("event_logger", getattr(instance, "_event_logger", None))
    use_legacy_attributes = kwargs.pop("use_legacy_attributes", getattr(instance, "config", Config()).use_legacy_attributes)

    try:
        _handle_request(span, args, kwargs, llm_model, event_logger, use_legacy_attributes)
        response = await wrapped(*args, **kwargs)
        
        if response:
            if is_async_streaming_response(response):
                return _abuild_from_streaming_response(span, response, llm_model, event_logger, use_legacy_attributes)
            else:
                _handle_response(span, response, llm_model, event_logger, use_legacy_attributes)
        
        span.set_status(Status(StatusCode.OK))
        return response
    except Exception as ex:
        span.set_status(Status(StatusCode.ERROR))
        span.record_exception(ex)
        raise
    finally:
        if not is_async_streaming_response(response):
            span.end()


@_with_tracer_wrapper
def _wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    # Get model name with proper fallbacks
    llm_model = "unknown"
    if hasattr(instance, "_model_id"):
        llm_model = instance._model_id
    if hasattr(instance, "_model_name"):
        llm_model = instance._model_name.replace("publishers/google/models/", "")

    name = to_wrap.get("span_name")
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Gemini",
            SpanAttributes.LLM_REQUEST_MODEL: llm_model,
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
    )

    # Get event logger and config from kwargs if provided, otherwise from instance
    event_logger = kwargs.pop("event_logger", getattr(instance, "_event_logger", None))
    use_legacy_attributes = kwargs.pop("use_legacy_attributes", getattr(instance, "config", Config()).use_legacy_attributes)

    try:
        _handle_request(span, args, kwargs, llm_model, event_logger, use_legacy_attributes)
        response = wrapped(*args, **kwargs)
        
        if response:
            if is_streaming_response(response):
                return _build_from_streaming_response(span, response, llm_model, event_logger, use_legacy_attributes)
            else:
                _handle_response(span, response, llm_model, event_logger, use_legacy_attributes)
        
        span.set_status(Status(StatusCode.OK))
        return response
    except Exception as ex:
        span.set_status(Status(StatusCode.ERROR))
        span.record_exception(ex)
        raise
    finally:
        if not is_streaming_response(response):
            span.end()


class GoogleGenerativeAiInstrumentor(BaseInstrumentor):
    """An instrumentor for Google Generative AI's client library."""

    def __init__(self, config: Config = None):
        super().__init__()
        self.config = config or Config()
        self._tracer = None
        self._event_logger = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        self._tracer = get_tracer(__name__, __version__)
        
        # Get event logger from kwargs or create new one
        self._event_logger = kwargs.get("event_logger", None)
        
        for wrapped_method in WRAPPED_METHODS:
            wrap_function_wrapper(
                wrapped_method["package"],
                f"{wrapped_method['object']}.{wrapped_method['method']}",
                _wrap if not wrapped_method["method"].endswith("_async") else _awrap,
            )

    def _uninstrument(self, **kwargs):
        """Removes instrumentation from Google Generative AI client library."""
        for wrapped_method in WRAPPED_METHODS:
            unwrap(
                f"{wrapped_method['package']}.{wrapped_method['object']}",
                wrapped_method["method"],
            )
        self._tracer = None
        self._event_logger = None
