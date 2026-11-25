"""OpenTelemetry Together AI instrumentation"""

import logging
import time
import warnings
from typing import Callable, Collection, Optional

from opentelemetry import context as context_api
from opentelemetry._logs import get_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.together.config import Config
from opentelemetry.instrumentation.together.event_emitter import (
    emit_completion_event,
    emit_prompt_events,
)
from opentelemetry.instrumentation.together.span_utils import (
    set_input_attributes,
    set_model_input_attributes,
    set_model_response_attributes,
    set_output_attributes,
)
from opentelemetry.instrumentation.together.utils import (
    TOGETHER_PROVIDER,
    dont_throw,
    is_metrics_enabled,
    model_as_dict,
    should_emit_events,
)
from opentelemetry.instrumentation.together.version import __version__
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.metrics import get_meter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics as GenAIMetrics
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
)
from opentelemetry.trace import SpanKind, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("together >= 1.2.0, <4",)

_CHAT_MODULE = "together.resources.chat.completions"
_COMPLETIONS_MODULE = "together.resources.completions"

_OPERATION_NAMES = {
    LLMRequestTypeValues.CHAT: GenAIAttributes.GenAiOperationNameValues.CHAT.value,
    LLMRequestTypeValues.COMPLETION: GenAIAttributes.GenAiOperationNameValues.TEXT_COMPLETION.value,
}


def _method(module, obj, request_type, is_async):
    return {
        "module": module,
        "object": obj,
        "request_type": request_type,
        "operation": _OPERATION_NAMES[request_type],
        "is_async": is_async,
    }


_CHAT = LLMRequestTypeValues.CHAT
_COMPLETION = LLMRequestTypeValues.COMPLETION

# together 2.x/3.x (Stainless SDKs) sets this request header on calls made through
# `.with_raw_response` ("raw") and `.with_streaming_response` ("stream"). `create`
# then returns an APIResponse wrapper instead of the response model.
_RAW_RESPONSE_HEADER = "X-Stainless-Raw-Response"

# together 1.x and together 2.x/3.x use the same modules but different class
# names. Every entry is tried; the ones missing from the installed version are
# skipped in `_instrument`.
WRAPPED_METHODS = [
    # together 1.x
    _method(_CHAT_MODULE, "ChatCompletions", _CHAT, is_async=False),
    _method(_CHAT_MODULE, "AsyncChatCompletions", _CHAT, is_async=True),
    _method(_COMPLETIONS_MODULE, "Completions", _COMPLETION, is_async=False),
    _method(_COMPLETIONS_MODULE, "AsyncCompletions", _COMPLETION, is_async=True),
    # together 2.x / 3.x
    _method(_CHAT_MODULE, "CompletionsResource", _CHAT, is_async=False),
    _method(_CHAT_MODULE, "AsyncCompletionsResource", _CHAT, is_async=True),
    _method(_COMPLETIONS_MODULE, "CompletionsResource", _COMPLETION, is_async=False),
    _method(_COMPLETIONS_MODULE, "AsyncCompletionsResource", _COMPLETION, is_async=True),
]


class _Telemetry:
    """The signal producers shared by every wrapped method."""

    def __init__(self, tracer, event_logger, token_histogram, duration_histogram):
        self.tracer = tracer
        self.event_logger = event_logger
        self.token_histogram = token_histogram
        self.duration_histogram = duration_histogram


def _with_telemetry_wrapper(func):
    """Helper for providing the telemetry producers to wrapper functions."""

    def _with_telemetry(telemetry, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(telemetry, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_telemetry


def _is_suppressed():
    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    )


@dont_throw
def _handle_input(span, event_logger, llm_request_type, kwargs):
    set_model_input_attributes(span, kwargs)

    if should_emit_events() and event_logger:
        emit_prompt_events(event_logger, llm_request_type, kwargs)
    else:
        set_input_attributes(span, llm_request_type, kwargs)


@dont_throw
def _handle_response(span, event_logger, llm_request_type, response):
    """`response` is the response model already converted with `model_as_dict`."""
    set_model_response_attributes(span, response)

    if should_emit_events() and event_logger:
        emit_completion_event(event_logger, llm_request_type, response)
    else:
        set_output_attributes(span, llm_request_type, response)


def _raw_response_mode(kwargs):
    return (kwargs.get("extra_headers") or {}).get(_RAW_RESPONSE_HEADER)


@dont_throw
def _should_parse_raw_response(response, kwargs):
    # Only `with_raw_response` responses are parsed: their body has already been
    # read, and APIResponse caches what `parse()` returns, so the app's own
    # `parse()` call is unaffected. A `with_streaming_response` body is still
    # unread when `create` returns, and reading it here would consume it before
    # the app does.
    return _raw_response_mode(kwargs) == "raw" and not kwargs.get("stream") and hasattr(response, "parse")


@dont_throw
def _parse_raw_response(response):
    return response.parse()


async def _aparse_raw_response(response):
    try:
        return await response.parse()
    except Exception as e:
        logger.debug("OpenLLMetry failed to parse the together raw response: %s", e)
        if Config.exception_logger:
            Config.exception_logger(e)
        return None


@dont_throw
def _response_as_dict(response):
    """The response model as a dict, or None when there is no model to read (e.g. an
    unread `with_streaming_response` body)."""
    response = model_as_dict(response)
    return response if isinstance(response, dict) else None


def _metric_attributes(to_wrap, kwargs, response=None):
    attributes = {
        **Config.get_common_metrics_attributes(),
        GenAIAttributes.GEN_AI_PROVIDER_NAME: TOGETHER_PROVIDER,
        GenAIAttributes.GEN_AI_OPERATION_NAME: to_wrap.get("operation"),
        GenAIAttributes.GEN_AI_REQUEST_MODEL: kwargs.get("model"),
    }
    if response is not None:
        attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] = response.get("model")
    return {key: value for key, value in attributes.items() if value is not None}


@dont_throw
def _record_duration(telemetry, to_wrap, kwargs, start_time, response=None, error=None):
    if telemetry.duration_histogram is None:
        return
    attributes = _metric_attributes(to_wrap, kwargs, response)
    if error is not None:
        attributes[ERROR_TYPE] = error.__class__.__name__
    telemetry.duration_histogram.record(time.perf_counter() - start_time, attributes=attributes)


@dont_throw
def _record_token_usage(telemetry, to_wrap, kwargs, response):
    if telemetry.token_histogram is None:
        return
    usage = response.get("usage") or {}
    attributes = _metric_attributes(to_wrap, kwargs, response)
    for token_type, key in (("input", "prompt_tokens"), ("output", "completion_tokens")):
        tokens = usage.get(key)
        if isinstance(tokens, int) and tokens >= 0:
            telemetry.token_histogram.record(
                tokens,
                attributes={**attributes, GenAIAttributes.GEN_AI_TOKEN_TYPE: token_type},
            )


def _start_span(telemetry, to_wrap, kwargs):
    operation = to_wrap.get("operation")
    model = kwargs.get("model")
    attributes = {
        GenAIAttributes.GEN_AI_PROVIDER_NAME: TOGETHER_PROVIDER,
        GenAIAttributes.GEN_AI_OPERATION_NAME: operation,
    }
    if model:
        attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] = model
    span = telemetry.tracer.start_span(
        f"{operation} {model}" if model else operation,
        kind=SpanKind.CLIENT,
        attributes=attributes,
    )
    _handle_input(span, telemetry.event_logger, to_wrap.get("request_type"), kwargs)
    return span


def _end_span_with_error(telemetry, to_wrap, kwargs, start_time, span, error):
    _record_duration(telemetry, to_wrap, kwargs, start_time, error=error)
    span.set_attribute(ERROR_TYPE, error.__class__.__name__)
    span.record_exception(error)
    span.set_status(Status(StatusCode.ERROR, str(error)))
    span.end()


def _end_span_with_response(telemetry, to_wrap, kwargs, start_time, span, response, parsed):
    """`parsed` is the response model to read telemetry from: `response` itself, or
    what a `with_raw_response` wrapper parsed to."""
    # Streaming responses are not instrumented yet: the span only covers the
    # request, and there is no complete response to read attributes or usage from.
    if response is not None and not kwargs.get("stream"):
        response_dict = _response_as_dict(parsed)
        _record_duration(telemetry, to_wrap, kwargs, start_time, response=response_dict)
        if response_dict:
            _record_token_usage(telemetry, to_wrap, kwargs, response_dict)
            _handle_response(span, telemetry.event_logger, to_wrap.get("request_type"), response_dict)
    if response and span.is_recording():
        span.set_status(Status(StatusCode.OK))
    span.end()


@_with_telemetry_wrapper
def _wrap(telemetry, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every sync function defined in WRAPPED_METHODS."""
    if _is_suppressed():
        return wrapped(*args, **kwargs)

    span = _start_span(telemetry, to_wrap, kwargs)
    start_time = time.perf_counter()
    try:
        response = wrapped(*args, **kwargs)
    except BaseException as e:
        _end_span_with_error(telemetry, to_wrap, kwargs, start_time, span, e)
        raise

    parsed = _parse_raw_response(response) if _should_parse_raw_response(response, kwargs) else response
    _end_span_with_response(telemetry, to_wrap, kwargs, start_time, span, response, parsed)
    return response


@_with_telemetry_wrapper
async def _awrap(telemetry, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every async function defined in WRAPPED_METHODS."""
    if _is_suppressed():
        return await wrapped(*args, **kwargs)

    span = _start_span(telemetry, to_wrap, kwargs)
    start_time = time.perf_counter()
    try:
        response = await wrapped(*args, **kwargs)
    except BaseException as e:
        # BaseException so the span also ends on asyncio cancellation.
        _end_span_with_error(telemetry, to_wrap, kwargs, start_time, span, e)
        raise

    parsed = await _aparse_raw_response(response) if _should_parse_raw_response(response, kwargs) else response
    _end_span_with_response(telemetry, to_wrap, kwargs, start_time, span, response, parsed)
    return response


class TogetherAiInstrumentor(BaseInstrumentor):
    """An instrumentor for Together AI's client library."""

    def __init__(
        self,
        exception_logger=None,
        use_attributes: Optional[bool] = None,
        use_legacy_attributes: Optional[bool] = None,
        get_common_metrics_attributes: Callable[[], dict] = lambda: {},
    ):
        super().__init__()
        if use_attributes is not None and use_legacy_attributes is not None:
            raise TypeError(
                "Cannot pass both `use_attributes` and `use_legacy_attributes`; "
                "`use_legacy_attributes` is deprecated, use `use_attributes` instead."
            )
        if use_legacy_attributes is not None:
            warnings.warn(
                "`use_legacy_attributes` is deprecated and will be removed in a "
                "future release; use `use_attributes` instead. The current OTel "
                "GenAI spec emits prompts/completions as span attributes "
                "(`gen_ai.input.messages` / `gen_ai.output.messages`), which is "
                "what `use_attributes=True` (the default) does. "
                "`use_attributes=False` opts into the events path instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            use_attributes = use_legacy_attributes
        if use_attributes is None:
            use_attributes = True
        Config.exception_logger = exception_logger
        Config.get_common_metrics_attributes = get_common_metrics_attributes
        Config.use_legacy_attributes = use_attributes

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        token_histogram = None
        duration_histogram = None
        if is_metrics_enabled():
            meter = get_meter(__name__, __version__, kwargs.get("meter_provider"))
            token_histogram = GenAIMetrics.create_gen_ai_client_token_usage(meter)
            duration_histogram = GenAIMetrics.create_gen_ai_client_operation_duration(meter)

        event_logger = None
        if not Config.use_legacy_attributes:
            logger_provider = kwargs.get("logger_provider")
            event_logger = get_logger(
                __name__, __version__, logger_provider=logger_provider
            )

        telemetry = _Telemetry(tracer, event_logger, token_histogram, duration_histogram)

        for wrapped_method in WRAPPED_METHODS:
            module = wrapped_method.get("module")
            obj = wrapped_method.get("object")
            wrapper = _awrap if wrapped_method.get("is_async") else _wrap
            try:
                wrap_function_wrapper(
                    module,
                    f"{obj}.create",
                    wrapper(telemetry, wrapped_method),
                )
            except (ImportError, AttributeError):
                # Class isn't in the installed together version.
                logger.debug("Skipping %s.%s.create", module, obj)

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            module = wrapped_method.get("module")
            obj = wrapped_method.get("object")
            try:
                unwrap(f"{module}.{obj}", "create")
            except ImportError:
                # Class isn't in the installed together version.
                pass
