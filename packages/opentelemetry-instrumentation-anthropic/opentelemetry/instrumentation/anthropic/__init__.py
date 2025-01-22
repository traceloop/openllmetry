"""OpenTelemetry Anthropic instrumentation"""

import json
import logging
import os
import time
from typing import Callable, Collection, Dict, Any, Optional
from typing_extensions import Coroutine

from anthropic._streaming import AsyncStream, Stream
from opentelemetry import context as context_api
from opentelemetry.instrumentation.anthropic.config import Config
from opentelemetry.instrumentation.anthropic.events import (
    create_prompt_event,
    create_completion_event,
    create_tool_call_event,
)
from opentelemetry.instrumentation.anthropic.streaming import (
    abuild_from_streaming_response,
    build_from_streaming_response,
)
from opentelemetry.instrumentation.anthropic.utils import (
    acount_prompt_tokens_from_request,
    dont_throw,
    error_metrics_attributes,
    count_prompt_tokens_from_request,
    run_async,
    set_span_attribute,
    shared_metrics_attributes,
    should_send_prompts,
)
from opentelemetry.instrumentation.anthropic.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap
from opentelemetry.metrics import Counter, Histogram, Meter, get_meter
from opentelemetry._events import EventLogger, get_event_logger
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    SpanAttributes,
    Meters,
)
from opentelemetry.trace import SpanKind, Tracer, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("anthropic >= 0.3.11",)

WRAPPED_METHODS = [
    {
        "package": "anthropic.resources.completions",
        "object": "Completions",
        "method": "create",
        "span_name": "anthropic.completion",
    },
    {
        "package": "anthropic.resources.messages",
        "object": "Messages",
        "method": "create",
        "span_name": "anthropic.chat",
    },
    {
        "package": "anthropic.resources.messages",
        "object": "Messages",
        "method": "stream",
        "span_name": "anthropic.chat",
    },
    {
        "package": "anthropic.resources.beta.prompt_caching.messages",
        "object": "Messages",
        "method": "create",
        "span_name": "anthropic.chat",
    },
    {
        "package": "anthropic.resources.beta.prompt_caching.messages",
        "object": "Messages",
        "method": "stream",
        "span_name": "anthropic.chat",
    },
]
WRAPPED_AMETHODS = [
    {
        "package": "anthropic.resources.completions",
        "object": "AsyncCompletions",
        "method": "create",
        "span_name": "anthropic.completion",
    },
    {
        "package": "anthropic.resources.messages",
        "object": "AsyncMessages",
        "method": "create",
        "span_name": "anthropic.chat",
    },
    {
        "package": "anthropic.resources.messages",
        "object": "AsyncMessages",
        "method": "stream",
        "span_name": "anthropic.chat",
    },
    {
        "package": "anthropic.resources.beta.prompt_caching.messages",
        "object": "AsyncMessages",
        "method": "create",
        "span_name": "anthropic.chat",
    },
    {
        "package": "anthropic.resources.beta.prompt_caching.messages",
        "object": "AsyncMessages",
        "method": "stream",
        "span_name": "anthropic.chat",
    },
]


def is_streaming_response(response):
    return isinstance(response, Stream) or isinstance(response, AsyncStream)


async def _process_image_item(item, trace_id, span_id, message_index, content_index):
    if not Config.upload_base64_image:
        return item

    image_format = item.get("source").get("media_type").split("/")[1]
    image_name = f"message_{message_index}_content_{content_index}.{image_format}"
    base64_string = item.get("source").get("data")
    url = await Config.upload_base64_image(trace_id, span_id, image_name, base64_string)

    return {"type": "image_url", "image_url": {"url": url}}


async def _dump_content(message_index, content, span):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # If the content is a list of text blocks, concatenate them.
        # This is more commonly used in prompt caching.
        if all([item.get("type") == "text" for item in content]):
            return "".join([item.get("text") for item in content])

        content = [
            (
                await _process_image_item(
                    item, span.context.trace_id, span.context.span_id, message_index, j
                )
                if _is_base64_image(item)
                else item
            )
            for j, item in enumerate(content)
        ]

        return json.dumps(content)


@dont_throw
async def _aset_input_attributes(span, kwargs, event_logger=None):
    # Always set basic attributes regardless of mode
    set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
    set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_tokens_to_sample")
    )
    set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TEMPERATURE, kwargs.get("temperature")
    )
    set_span_attribute(span, SpanAttributes.LLM_REQUEST_TOP_P, kwargs.get("top_p"))
    set_span_attribute(
        span, SpanAttributes.LLM_FREQUENCY_PENALTY, kwargs.get("frequency_penalty")
    )
    set_span_attribute(
        span, SpanAttributes.LLM_PRESENCE_PENALTY, kwargs.get("presence_penalty")
    )
    set_span_attribute(span, SpanAttributes.LLM_IS_STREAMING, kwargs.get("stream"))

    if should_send_prompts():
        if kwargs.get("prompt") is not None:
            # Legacy attribute-based approach
            if Config.use_legacy_attributes:
                set_span_attribute(
                    span, f"{SpanAttributes.LLM_PROMPTS}.0.user", kwargs.get("prompt")
                )
            # Event-based approach
            if event_logger and not Config.use_legacy_attributes:
                event_logger.emit(
                    create_prompt_event(
                        content=kwargs.get("prompt"),
                        role="user",
                        span_ctx=span.get_span_context(),
                    )
                )

        elif kwargs.get("messages") is not None:
            has_system_message = False
            system_content = None
            
            if kwargs.get("system"):
                has_system_message = True
                system_content = await _dump_content(
                    message_index=0, span=span, content=kwargs.get("system")
                )
                # Legacy attribute-based approach
                if Config.use_legacy_attributes:
                    set_span_attribute(
                        span,
                        f"{SpanAttributes.LLM_PROMPTS}.0.content",
                        system_content,
                    )
                    set_span_attribute(
                        span,
                        f"{SpanAttributes.LLM_PROMPTS}.0.role",
                        "system",
                    )

            for i, message in enumerate(kwargs.get("messages")):
                prompt_index = i + (1 if has_system_message else 0)
                content = await _dump_content(
                    message_index=i, span=span, content=message.get("content")
                )
                
                # Legacy attribute-based approach
                if Config.use_legacy_attributes:
                    set_span_attribute(
                        span,
                        f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.content",
                        content,
                    )
                    set_span_attribute(
                        span,
                        f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.role",
                        message.get("role"),
                    )
                
                # Event-based approach
                if event_logger and not Config.use_legacy_attributes:
                    event_logger.emit(
                        create_prompt_event(
                            content=content,
                            role=message.get("role"),
                            span_ctx=span.get_span_context(),
                            system=system_content if i == 0 and has_system_message else None,
                            index=prompt_index,
                        )
                    )

        if kwargs.get("tools") is not None:
            for i, tool in enumerate(kwargs.get("tools")):
                # Legacy attribute-based approach
                if Config.use_legacy_attributes:
                    prefix = f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}"
                    set_span_attribute(span, f"{prefix}.name", tool.get("name"))
                    set_span_attribute(span, f"{prefix}.description", tool.get("description"))
                    input_schema = tool.get("input_schema")
                    if input_schema is not None:
                        set_span_attribute(span, f"{prefix}.input_schema", json.dumps(input_schema))
                
                # Event-based approach
                if event_logger and not Config.use_legacy_attributes:
                    event_logger.emit(
                        create_tool_call_event(
                            tool_call={
                                "name": tool.get("name"),
                                "description": tool.get("description"),
                                "input_schema": tool.get("input_schema"),
                            },
                            span_ctx=span.get_span_context(),
                            index=i,
                        )
                    )


def _set_span_completions(span, response, event_logger=None):
    index = 0
    
    # Legacy attribute-based approach
    if Config.use_legacy_attributes:
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
        set_span_attribute(span, f"{prefix}.finish_reason", response.get("stop_reason"))
        if response.get("role"):
            set_span_attribute(span, f"{prefix}.role", response.get("role"))

        if response.get("completion"):
            set_span_attribute(span, f"{prefix}.content", response.get("completion"))
        elif response.get("content"):
            tool_call_index = 0
            text = ""
            tool_calls = []
            
            for content in response.get("content"):
                content_block_type = content.type
                # usually, Anthropic responds with just one text block,
                # but the API allows for multiple text blocks, so concatenate them
                if content_block_type == "text":
                    text += content.text
                elif content_block_type == "tool_use":
                    content = dict(content)
                    tool_calls.append(content)
                    set_span_attribute(
                        span,
                        f"{prefix}.tool_calls.{tool_call_index}.id",
                        content.get("id"),
                    )
                    set_span_attribute(
                        span,
                        f"{prefix}.tool_calls.{tool_call_index}.name",
                        content.get("name"),
                    )
                    tool_arguments = content.get("input")
                    if tool_arguments is not None:
                        set_span_attribute(
                            span,
                            f"{prefix}.tool_calls.{tool_call_index}.arguments",
                            json.dumps(tool_arguments),
                        )
                    tool_call_index += 1
            
            if text:
                set_span_attribute(span, f"{prefix}.content", text)
    
    # Event-based approach
    if event_logger and not Config.use_legacy_attributes:
        content = ""
        tool_calls = []
        
        if response.get("completion"):
            content = response.get("completion")
        elif response.get("content"):
            for content_block in response.get("content"):
                if content_block.type == "text":
                    content += content_block.text
                elif content_block.type == "tool_use":
                    tool_calls.append({
                        "id": content_block.get("id"),
                        "name": content_block.get("name"),
                        "arguments": json.dumps(content_block.get("input")) if content_block.get("input") else None
                    })
        
        event_logger.emit(
            create_completion_event(
                content=content,
                role=response.get("role", "assistant"),
                finish_reason=response.get("stop_reason"),
                span_ctx=span.get_span_context(),
                index=index,
                tool_calls=tool_calls if tool_calls else None
            )
        )


@dont_throw
async def _aset_token_usage(
    span,
    anthropic,
    request,
    response,
    metric_attributes: dict = {},
    token_histogram: Histogram = None,
    choice_counter: Counter = None,
):
    if not isinstance(response, dict):
        response = response.__dict__

    if usage := response.get("usage"):
        prompt_tokens = usage.input_tokens
    else:
        prompt_tokens = await acount_prompt_tokens_from_request(anthropic, request)

    if usage := response.get("usage"):
        cache_read_tokens = dict(usage).get("cache_read_input_tokens", 0)
    else:
        cache_read_tokens = 0

    if usage := response.get("usage"):
        cache_creation_tokens = dict(usage).get("cache_creation_input_tokens", 0)
    else:
        cache_creation_tokens = 0

    input_tokens = prompt_tokens + cache_read_tokens + cache_creation_tokens

    if token_histogram and type(input_tokens) is int and input_tokens >= 0:
        token_histogram.record(
            input_tokens,
            attributes={
                **metric_attributes,
                SpanAttributes.LLM_TOKEN_TYPE: "input",
            },
        )

    if usage := response.get("usage"):
        completion_tokens = usage.output_tokens
    else:
        completion_tokens = 0
        if hasattr(anthropic, "count_tokens"):
            if response.get("completion"):
                completion_tokens = await anthropic.count_tokens(response.get("completion"))
            elif response.get("content"):
                completion_tokens = await anthropic.count_tokens(
                    response.get("content")[0].text
                )

    if token_histogram and type(completion_tokens) is int and completion_tokens >= 0:
        token_histogram.record(
            completion_tokens,
            attributes={
                **metric_attributes,
                SpanAttributes.LLM_TOKEN_TYPE: "output",
            },
        )

    total_tokens = input_tokens + completion_tokens

    choices = 0
    if type(response.get("content")) is list:
        choices = len(response.get("content"))
    elif response.get("completion"):
        choices = 1

    if choices > 0 and choice_counter:
        choice_counter.add(
            choices,
            attributes={
                **metric_attributes,
                SpanAttributes.LLM_RESPONSE_STOP_REASON: response.get("stop_reason"),
            },
        )

    set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, input_tokens)
    set_span_attribute(
        span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens
    )
    set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)

    set_span_attribute(
        span, SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, cache_read_tokens
    )
    set_span_attribute(
        span, SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS, cache_creation_tokens
    )


@dont_throw
def _set_token_usage(
    span,
    anthropic,
    request,
    response,
    metric_attributes: dict = {},
    token_histogram: Histogram = None,
    choice_counter: Counter = None,
):
    if not isinstance(response, dict):
        response = response.__dict__

    if usage := response.get("usage"):
        prompt_tokens = usage.input_tokens
    else:
        prompt_tokens = count_prompt_tokens_from_request(anthropic, request)

    if usage := response.get("usage"):
        cache_read_tokens = dict(usage).get("cache_read_input_tokens", 0)
    else:
        cache_read_tokens = 0

    if usage := response.get("usage"):
        cache_creation_tokens = dict(usage).get("cache_creation_input_tokens", 0)
    else:
        cache_creation_tokens = 0

    input_tokens = prompt_tokens + cache_read_tokens + cache_creation_tokens

    if token_histogram and type(input_tokens) is int and input_tokens >= 0:
        token_histogram.record(
            input_tokens,
            attributes={
                **metric_attributes,
                SpanAttributes.LLM_TOKEN_TYPE: "input",
            },
        )

    if usage := response.get("usage"):
        completion_tokens = usage.output_tokens
    else:
        completion_tokens = 0
        if hasattr(anthropic, "count_tokens"):
            if response.get("completion"):
                completion_tokens = anthropic.count_tokens(response.get("completion"))
            elif response.get("content"):
                completion_tokens = anthropic.count_tokens(response.get("content")[0].text)

    if token_histogram and type(completion_tokens) is int and completion_tokens >= 0:
        token_histogram.record(
            completion_tokens,
            attributes={
                **metric_attributes,
                SpanAttributes.LLM_TOKEN_TYPE: "output",
            },
        )

    total_tokens = input_tokens + completion_tokens

    choices = 0
    if type(response.get("content")) is list:
        choices = len(response.get("content"))
    elif response.get("completion"):
        choices = 1

    if choices > 0 and choice_counter:
        choice_counter.add(
            choices,
            attributes={
                **metric_attributes,
                SpanAttributes.LLM_RESPONSE_STOP_REASON: response.get("stop_reason"),
            },
        )

    set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, input_tokens)
    set_span_attribute(
        span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens
    )
    set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)

    set_span_attribute(
        span, SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, cache_read_tokens
    )
    set_span_attribute(
        span, SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS, cache_creation_tokens
    )


@dont_throw
def _set_response_attributes(span, response):
    if not isinstance(response, dict):
        response = response.__dict__
    set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, response.get("model"))

    if response.get("usage"):
        prompt_tokens = response.get("usage").input_tokens
        completion_tokens = response.get("usage").output_tokens
        set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
        set_span_attribute(
            span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens
        )
        set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            prompt_tokens + completion_tokens,
        )

    if should_send_prompts():
        _set_span_completions(span, response)


def _create_metrics(meter: Meter):
    token_histogram = meter.create_histogram(
        name=Meters.LLM_TOKEN_USAGE,
        unit="token",
        description="Measures number of input and output tokens used",
    )

    choice_counter = meter.create_counter(
        name=Meters.LLM_GENERATION_CHOICES,
        unit="choice",
        description="Number of choices returned by chat completions call",
    )

    duration_histogram = meter.create_histogram(
        name=Meters.LLM_OPERATION_DURATION,
        unit="s",
        description="GenAI operation duration",
    )

    exception_counter = meter.create_counter(
        name=Meters.LLM_ANTHROPIC_COMPLETION_EXCEPTIONS,
        unit="time",
        description="Number of exceptions occurred during chat completions",
    )

    return token_histogram, choice_counter, duration_histogram, exception_counter


def _is_base64_image(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False

    if not isinstance(item.get("source"), dict):
        return False

    if item.get("type") != "image" or item["source"].get("type") != "base64":
        return False

    return True


def _wrap(tracer, event_logger, token_histogram, choice_counter):
    """Wrap sync methods."""

    def wrapper(wrapped, instance, args, kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        name = wrapped.__name__
        with tracer.start_as_current_span(
            name=f"anthropic.{name}",
            kind=SpanKind.CLIENT,
            attributes={
                SpanAttributes.LLM_VENDOR: "Anthropic",
                SpanAttributes.LLM_REQUEST_TYPE: (
                    LLMRequestTypeValues.COMPLETION.value
                    if name == "create"
                    else LLMRequestTypeValues.CHAT.value
                ),
            },
        ) as span:
            try:
                # Set input attributes
                run_async(_aset_input_attributes(span, kwargs, event_logger))

                # Call the original function
                response = wrapped(*args, **kwargs)

                # Handle streaming response
                if is_streaming_response(response):
                    return build_from_streaming_response(
                        response, span, event_logger, token_histogram, choice_counter
                    )

                # Set completion attributes
                _set_span_completions(span, response, event_logger)

                # Update metrics
                if Config.enrich_token_usage:
                    prompt_tokens = count_prompt_tokens_from_request(kwargs)
                    token_histogram.record(
                        prompt_tokens,
                        attributes={
                            **shared_metrics_attributes(),
                            Meters.LLM_REQUEST_TYPE: "prompt",
                        },
                    )
                    choice_counter.add(1, attributes=shared_metrics_attributes())

                return response

            except Exception as ex:
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(ex)
                if Config.exception_logger:
                    Config.exception_logger(ex)
                raise

    return wrapper


def _awrap(tracer, event_logger, token_histogram, choice_counter):
    """Wrap async methods."""

    async def wrapper(wrapped, instance, args, kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        name = wrapped.__name__
        with tracer.start_as_current_span(
            name=f"anthropic.{name}",
            kind=SpanKind.CLIENT,
            attributes={
                SpanAttributes.LLM_VENDOR: "Anthropic",
                SpanAttributes.LLM_REQUEST_TYPE: (
                    LLMRequestTypeValues.COMPLETION.value
                    if name == "create"
                    else LLMRequestTypeValues.CHAT.value
                ),
            },
        ) as span:
            try:
                # Set input attributes
                await _aset_input_attributes(span, kwargs, event_logger)

                # Call the original function
                response = await wrapped(*args, **kwargs)

                # Handle streaming response
                if is_streaming_response(response):
                    return await abuild_from_streaming_response(
                        response, span, event_logger, token_histogram, choice_counter
                    )

                # Set completion attributes
                _set_span_completions(span, response, event_logger)

                # Update metrics
                if Config.enrich_token_usage:
                    prompt_tokens = await acount_prompt_tokens_from_request(kwargs)
                    token_histogram.record(
                        prompt_tokens,
                        attributes={
                            **shared_metrics_attributes(),
                            Meters.LLM_REQUEST_TYPE: "prompt",
                        },
                    )
                    choice_counter.add(1, attributes=shared_metrics_attributes())

                return response

            except Exception as ex:
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(ex)
                if Config.exception_logger:
                    Config.exception_logger(ex)
                raise

    return wrapper


def is_metrics_enabled() -> bool:
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"


class AnthropicInstrumentor(BaseInstrumentor):
    """An instrumentor for Anthropic's client library."""

    def __init__(
        self,
        enrich_token_usage: bool = False,
        exception_logger=None,
        get_common_metrics_attributes: Callable[[], dict] = lambda: {},
        upload_base64_image: Optional[
            Callable[[str, str, str, str], Coroutine[None, None, str]]
        ] = None,
    ):
        super().__init__()
        Config.exception_logger = exception_logger
        Config.enrich_token_usage = enrich_token_usage
        Config.get_common_metrics_attributes = get_common_metrics_attributes
        Config.upload_base64_image = upload_base64_image
        self._tracer = None
        self._event_logger = None
        self._meter = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        # Initialize tracer, event logger and meter
        tracer_provider = kwargs.get("tracer_provider")
        self._tracer = get_tracer(__name__, __version__, tracer_provider)
        self._event_logger = get_event_logger(__name__, __version__)
        self._meter = get_meter(__name__, __version__)

        # Get configuration
        Config.use_legacy_attributes = kwargs.get("use_legacy_attributes", True)

        # meter and counters are inited here
        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        if is_metrics_enabled():
            (
                token_histogram,
                choice_counter,
                duration_histogram,
                exception_counter,
            ) = _create_metrics(meter)
        else:
            (
                token_histogram,
                choice_counter,
                duration_histogram,
                exception_counter,
            ) = (None, None, None, None)

        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            try:
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}",
                    _wrap(
                        self._tracer,
                        self._event_logger,
                        token_histogram,
                        choice_counter,
                    ),
                )
            except ModuleNotFoundError:
                pass  # that's ok, we don't want to fail if some methods do not exist

        for wrapped_method in WRAPPED_AMETHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            try:
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}",
                    _awrap(
                        self._tracer,
                        self._event_logger,
                        token_histogram,
                        choice_counter,
                    ),
                )
            except ModuleNotFoundError:
                pass  # that's ok, we don't want to fail if some methods do not exist

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"{wrap_package}.{wrap_object}",
                wrapped_method.get("method"),
            )
        for wrapped_method in WRAPPED_AMETHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"{wrap_package}.{wrap_object}",
                wrapped_method.get("method"),
            )
