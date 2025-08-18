"""OpenTelemetry Anthropic instrumentation"""

import logging
import os
import time
from typing import Callable, Collection, Optional

from opentelemetry import context as context_api
from opentelemetry._events import EventLogger, get_event_logger
from opentelemetry.instrumentation.anthropic.config import Config
from opentelemetry.instrumentation.anthropic.event_emitter import (
    emit_input_events,
    emit_response_events,
)
from opentelemetry.instrumentation.anthropic.span_utils import (
    aset_input_attributes,
    set_response_attributes,
)
from opentelemetry.instrumentation.anthropic.streaming import (
    abuild_from_streaming_response,
    build_from_streaming_response,
    WrappedAsyncMessageStreamManager,
    WrappedMessageStreamManager,
)
from opentelemetry.instrumentation.anthropic.utils import (
    acount_prompt_tokens_from_request,
    count_prompt_tokens_from_request,
    dont_throw,
    error_metrics_attributes,
    run_async,
    set_span_attribute,
    shared_metrics_attributes,
    should_emit_events,
)
from opentelemetry.instrumentation.anthropic.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap
from opentelemetry.metrics import Counter, Histogram, Meter, get_meter
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    Meters,
    SpanAttributes,
)
from opentelemetry.trace import Span, SpanKind, Tracer, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from typing_extensions import Coroutine
from wrapt import wrap_function_wrapper

from anthropic._streaming import AsyncStream, Stream

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
    # This method is on an async resource, but is meant to be called as
    # an async context manager (async with), which we don't need to await;
    # thus, we wrap it with a sync wrapper
    {
        "package": "anthropic.resources.messages",
        "object": "AsyncMessages",
        "method": "stream",
        "span_name": "anthropic.chat",
    },
    # Beta API methods (regular Anthropic SDK)
    {
        "package": "anthropic.resources.beta.messages.messages",
        "object": "Messages",
        "method": "create",
        "span_name": "anthropic.chat",
    },
    {
        "package": "anthropic.resources.beta.messages.messages",
        "object": "Messages",
        "method": "stream",
        "span_name": "anthropic.chat",
    },
    # Beta API methods (Bedrock SDK)
    {
        "package": "anthropic.lib.bedrock._beta_messages",
        "object": "Messages",
        "method": "create",
        "span_name": "anthropic.chat",
    },
    {
        "package": "anthropic.lib.bedrock._beta_messages",
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
    # Beta API async methods (regular Anthropic SDK)
    {
        "package": "anthropic.resources.beta.messages.messages",
        "object": "AsyncMessages",
        "method": "create",
        "span_name": "anthropic.chat",
    },
    {
        "package": "anthropic.resources.beta.messages.messages",
        "object": "AsyncMessages",
        "method": "stream",
        "span_name": "anthropic.chat",
    },
    # Beta API async methods (Bedrock SDK)
    {
        "package": "anthropic.lib.bedrock._beta_messages",
        "object": "AsyncMessages",
        "method": "create",
        "span_name": "anthropic.chat",
    },
    {
        "package": "anthropic.lib.bedrock._beta_messages",
        "object": "AsyncMessages",
        "method": "stream",
        "span_name": "anthropic.chat",
    },
]


def is_streaming_response(response):
    return isinstance(response, Stream) or isinstance(response, AsyncStream)


def is_stream_manager(response):
    """Check if response is a MessageStreamManager or AsyncMessageStreamManager"""
    try:
        from anthropic.lib.streaming._messages import (
            MessageStreamManager,
            AsyncMessageStreamManager,
        )

        return isinstance(response, (MessageStreamManager, AsyncMessageStreamManager))
    except ImportError:
        # Check by class name as fallback
        return (
            response.__class__.__name__ == "MessageStreamManager"
            or response.__class__.__name__ == "AsyncMessageStreamManager"
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
    import inspect

    # If we get a coroutine, await it
    if inspect.iscoroutine(response):
        try:
            response = await response
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to await coroutine response: {e}")
            return

    # Handle with_raw_response wrapped responses first
    if response and hasattr(response, "parse") and callable(response.parse):
        try:
            response = response.parse()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to parse with_raw_response: {e}")
            return

    # Safely get usage attribute without extracting the whole object
    usage = getattr(response, "usage", None) if response else None

    if usage:
        prompt_tokens = getattr(usage, "input_tokens", 0)
        cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
    else:
        prompt_tokens = await acount_prompt_tokens_from_request(anthropic, request)
        cache_read_tokens = 0
        cache_creation_tokens = 0

    input_tokens = prompt_tokens + cache_read_tokens + cache_creation_tokens

    if token_histogram and isinstance(input_tokens, int) and input_tokens >= 0:
        token_histogram.record(
            input_tokens,
            attributes={
                **metric_attributes,
                SpanAttributes.LLM_TOKEN_TYPE: "input",
            },
        )

    if usage:
        completion_tokens = getattr(usage, "output_tokens", 0)
    else:
        completion_tokens = 0
        if hasattr(anthropic, "count_tokens"):
            completion_attr = getattr(response, "completion", None)
            content_attr = getattr(response, "content", None)
            if completion_attr:
                completion_tokens = await anthropic.count_tokens(completion_attr)
            elif content_attr and len(content_attr) > 0:
                completion_tokens = await anthropic.count_tokens(
                    content_attr[0].text
                )

    if (
        token_histogram
        and isinstance(completion_tokens, int)
        and completion_tokens >= 0
    ):
        token_histogram.record(
            completion_tokens,
            attributes={
                **metric_attributes,
                SpanAttributes.LLM_TOKEN_TYPE: "output",
            },
        )

    total_tokens = input_tokens + completion_tokens

    choices = 0
    content_attr = getattr(response, "content", None)
    completion_attr = getattr(response, "completion", None)
    if isinstance(content_attr, list):
        choices = len(content_attr)
    elif completion_attr:
        choices = 1

    if choices > 0 and choice_counter:
        choice_counter.add(
            choices,
            attributes={
                **metric_attributes,
                SpanAttributes.LLM_RESPONSE_STOP_REASON: getattr(response, "stop_reason", None),
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
        span,
        SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
        cache_creation_tokens,
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
    import inspect

    # If we get a coroutine, we cannot process it in sync context
    if inspect.iscoroutine(response):
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"_set_token_usage received coroutine {response} - token usage processing skipped")
        return

    # Handle with_raw_response wrapped responses first
    if response and hasattr(response, "parse") and callable(response.parse):
        try:
            response = response.parse()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to parse with_raw_response: {e}")
            return

    # Safely get usage attribute without extracting the whole object
    usage = getattr(response, "usage", None) if response else None

    if usage:
        prompt_tokens = getattr(usage, "input_tokens", 0)
        cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
    else:
        prompt_tokens = count_prompt_tokens_from_request(anthropic, request)
        cache_read_tokens = 0
        cache_creation_tokens = 0

    input_tokens = prompt_tokens + cache_read_tokens + cache_creation_tokens

    if token_histogram and isinstance(input_tokens, int) and input_tokens >= 0:
        token_histogram.record(
            input_tokens,
            attributes={
                **metric_attributes,
                SpanAttributes.LLM_TOKEN_TYPE: "input",
            },
        )

    if usage:
        completion_tokens = getattr(usage, "output_tokens", 0)
    else:
        completion_tokens = 0
        if hasattr(anthropic, "count_tokens"):
            completion_attr = getattr(response, "completion", None)
            content_attr = getattr(response, "content", None)
            if completion_attr:
                completion_tokens = anthropic.count_tokens(completion_attr)
            elif content_attr and len(content_attr) > 0:
                completion_tokens = anthropic.count_tokens(
                    content_attr[0].text
                )

    if (
        token_histogram
        and isinstance(completion_tokens, int)
        and completion_tokens >= 0
    ):
        token_histogram.record(
            completion_tokens,
            attributes={
                **metric_attributes,
                SpanAttributes.LLM_TOKEN_TYPE: "output",
            },
        )

    total_tokens = input_tokens + completion_tokens

    choices = 0
    content_attr = getattr(response, "content", None)
    completion_attr = getattr(response, "completion", None)
    if isinstance(content_attr, list):
        choices = len(content_attr)
    elif completion_attr:
        choices = 1

    if choices > 0 and choice_counter:
        choice_counter.add(
            choices,
            attributes={
                **metric_attributes,
                SpanAttributes.LLM_RESPONSE_STOP_REASON: getattr(response, "stop_reason", None),
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
        span,
        SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
        cache_creation_tokens,
    )


def _with_chat_telemetry_wrapper(func):
    """Helper for providing tracer for wrapper functions. Includes metric collectors."""

    def _with_chat_telemetry(
        tracer,
        token_histogram,
        choice_counter,
        duration_histogram,
        exception_counter,
        event_logger,
        to_wrap,
    ):
        def wrapper(wrapped, instance, args, kwargs):
            return func(
                tracer,
                token_histogram,
                choice_counter,
                duration_histogram,
                exception_counter,
                event_logger,
                to_wrap,
                wrapped,
                instance,
                args,
                kwargs,
            )

        return wrapper

    return _with_chat_telemetry


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


@dont_throw
def _handle_input(span: Span, event_logger: Optional[EventLogger], kwargs):
    if should_emit_events() and event_logger:
        emit_input_events(event_logger, kwargs)
    else:
        if not span.is_recording():
            return
        run_async(aset_input_attributes(span, kwargs))


@dont_throw
async def _ahandle_input(span: Span, event_logger: Optional[EventLogger], kwargs):
    if should_emit_events() and event_logger:
        emit_input_events(event_logger, kwargs)
    else:
        if not span.is_recording():
            return
        await aset_input_attributes(span, kwargs)


@dont_throw
async def _ahandle_response(span: Span, event_logger: Optional[EventLogger], response):
    if should_emit_events():
        emit_response_events(event_logger, response)
    else:
        if not span.is_recording():
            return
        from opentelemetry.instrumentation.anthropic.span_utils import (
            aset_response_attributes,
        )

        await aset_response_attributes(span, response)


@dont_throw
def _handle_response(span: Span, event_logger: Optional[EventLogger], response):
    if should_emit_events():
        emit_response_events(event_logger, response)
    else:
        if not span.is_recording():
            return
        set_response_attributes(span, response)


@_with_chat_telemetry_wrapper
def _wrap(
    tracer: Tracer,
    token_histogram: Histogram,
    choice_counter: Counter,
    duration_histogram: Histogram,
    exception_counter: Counter,
    event_logger: Optional[EventLogger],
    to_wrap,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Anthropic",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
    )

    _handle_input(span, event_logger, kwargs)

    start_time = time.time()
    try:
        response = wrapped(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        end_time = time.time()
        attributes = error_metrics_attributes(e)

        if duration_histogram:
            duration = end_time - start_time
            duration_histogram.record(duration, attributes=attributes)

        if exception_counter:
            exception_counter.add(1, attributes=attributes)

        raise e

    end_time = time.time()

    if is_streaming_response(response):
        return build_from_streaming_response(
            span,
            response,
            instance._client,
            start_time,
            token_histogram,
            choice_counter,
            duration_histogram,
            exception_counter,
            event_logger,
            kwargs,
        )
    elif is_stream_manager(response):
        if response.__class__.__name__ == "AsyncMessageStreamManager":
            return WrappedAsyncMessageStreamManager(
                response,
                span,
                instance._client,
                start_time,
                token_histogram,
                choice_counter,
                duration_histogram,
                exception_counter,
                event_logger,
                kwargs,
            )
        else:
            return WrappedMessageStreamManager(
                response,
                span,
                instance._client,
                start_time,
                token_histogram,
                choice_counter,
                duration_histogram,
                exception_counter,
                event_logger,
                kwargs,
            )
    elif response:
        try:
            metric_attributes = shared_metrics_attributes(response)

            if duration_histogram:
                duration = time.time() - start_time
                duration_histogram.record(
                    duration,
                    attributes=metric_attributes,
                )

            _handle_response(span, event_logger, response)
            if span.is_recording():
                _set_token_usage(
                    span,
                    instance._client,
                    kwargs,
                    response,
                    metric_attributes,
                    token_histogram,
                    choice_counter,
                )
        except Exception as ex:  # pylint: disable=broad-except
            logger.warning(
                "Failed to set response attributes for anthropic span, error: %s",
                str(ex),
            )

        if span.is_recording():
            span.set_status(Status(StatusCode.OK))
    span.end()
    return response


@_with_chat_telemetry_wrapper
async def _awrap(
    tracer,
    token_histogram: Histogram,
    choice_counter: Counter,
    duration_histogram: Histogram,
    exception_counter: Counter,
    event_logger: Optional[EventLogger],
    to_wrap,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return await wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Anthropic",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
    )
    await _ahandle_input(span, event_logger, kwargs)

    start_time = time.time()
    try:
        response = await wrapped(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        end_time = time.time()
        attributes = error_metrics_attributes(e)

        if duration_histogram:
            duration = end_time - start_time
            duration_histogram.record(duration, attributes=attributes)

        if exception_counter:
            exception_counter.add(1, attributes=attributes)

        raise e

    if is_streaming_response(response):
        return abuild_from_streaming_response(
            span,
            response,
            instance._client,
            start_time,
            token_histogram,
            choice_counter,
            duration_histogram,
            exception_counter,
            event_logger,
            kwargs,
        )
    elif is_stream_manager(response):
        if response.__class__.__name__ == "AsyncMessageStreamManager":
            return WrappedAsyncMessageStreamManager(
                response,
                span,
                instance._client,
                start_time,
                token_histogram,
                choice_counter,
                duration_histogram,
                exception_counter,
                event_logger,
                kwargs,
            )
        else:
            return WrappedMessageStreamManager(
                response,
                span,
                instance._client,
                start_time,
                token_histogram,
                choice_counter,
                duration_histogram,
                exception_counter,
                event_logger,
                kwargs,
            )
    elif response:
        from opentelemetry.instrumentation.anthropic.utils import (
            ashared_metrics_attributes,
        )

        metric_attributes = await ashared_metrics_attributes(response)

        if duration_histogram:
            duration = time.time() - start_time
            duration_histogram.record(
                duration,
                attributes=metric_attributes,
            )

        await _ahandle_response(span, event_logger, response)

        if span.is_recording():
            await _aset_token_usage(
                span,
                instance._client,
                kwargs,
                response,
                metric_attributes,
                token_histogram,
                choice_counter,
            )
            span.set_status(Status(StatusCode.OK))
    span.end()
    return response


def is_metrics_enabled() -> bool:
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"


class AnthropicInstrumentor(BaseInstrumentor):
    """An instrumentor for Anthropic's client library."""

    def __init__(
        self,
        enrich_token_usage: bool = False,
        exception_logger=None,
        use_legacy_attributes: bool = True,
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
        Config.use_legacy_attributes = use_legacy_attributes

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

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

        # event_logger is inited here
        event_logger = None

        if not Config.use_legacy_attributes:
            event_logger_provider = kwargs.get("event_logger_provider")
            event_logger = get_event_logger(
                __name__, __version__, event_logger_provider=event_logger_provider
            )

        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            try:
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}",
                    _wrap(
                        tracer,
                        token_histogram,
                        choice_counter,
                        duration_histogram,
                        exception_counter,
                        event_logger,
                        wrapped_method,
                    ),
                )
                logger.debug(
                    f"Successfully wrapped {wrap_package}.{wrap_object}.{wrap_method}"
                )
            except Exception as e:
                logger.debug(
                    f"Failed to wrap {wrap_package}.{wrap_object}.{wrap_method}: {e}"
                )
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
                        tracer,
                        token_histogram,
                        choice_counter,
                        duration_histogram,
                        exception_counter,
                        event_logger,
                        wrapped_method,
                    ),
                )
            except Exception:
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
