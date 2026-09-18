"""OpenTelemetry OCI Generative AI instrumentation"""

import logging
import os
import time
import warnings
from functools import partial
from typing import Callable, Collection, Optional

from opentelemetry import context as context_api
from opentelemetry._logs import get_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.oci_genai.config import Config
from opentelemetry.instrumentation.oci_genai.event_emitter import (
    emit_input_events,
    emit_response_events,
    emit_streaming_response_events,
)
from opentelemetry.instrumentation.oci_genai.span_utils import (
    CHAT,
    EMBEDDINGS,
    OCI_GENAI_PROVIDER,
    RERANK,
    TEXT_COMPLETION,
    get_request_model,
    is_stream_request,
    record_usage_metrics,
    set_input_attributes,
    set_output_attributes,
    set_request_attributes,
    set_response_attributes,
    set_streaming_output_attributes,
    set_streaming_response_attributes,
)
from opentelemetry.instrumentation.oci_genai.streaming import (
    OCIGenAIStreamWrapper,
    StreamAccumulator,
)
from opentelemetry.instrumentation.oci_genai.utils import dont_throw, should_emit_events
from opentelemetry.instrumentation.oci_genai.version import __version__
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap
from opentelemetry.metrics import Meter, get_meter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    Meters,
)
from opentelemetry.trace import SpanKind, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("oci >= 2.121.1",)

_CLIENT_MODULE = "oci.generative_ai_inference.generative_ai_inference_client"
_CLIENT_CLASS = "GenerativeAiInferenceClient"

WRAPPED_METHODS = [
    {"method": "chat", "details_kwarg": "chat_details", "operation": CHAT},
    {"method": "generate_text", "details_kwarg": "generate_text_details", "operation": TEXT_COMPLETION},
    {"method": "embed_text", "details_kwarg": "embed_text_details", "operation": EMBEDDINGS},
    {"method": "rerank_text", "details_kwarg": "rerank_text_details", "operation": RERANK},
]


def is_metrics_enabled() -> bool:
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"


def _create_metrics(meter: Meter):
    token_histogram = meter.create_histogram(
        name=Meters.LLM_TOKEN_USAGE,
        unit="token",
        description="Measures number of input and output tokens used",
    )

    duration_histogram = meter.create_histogram(
        name=Meters.LLM_OPERATION_DURATION,
        unit="s",
        description="GenAI operation duration",
    )

    return token_histogram, duration_histogram


def _span_name(operation, model):
    """Build span name per OTel semconv: '{operation_name} {model}'."""
    return f"{operation} {model}" if model else operation


def _resolve_details(args, kwargs, details_kwarg):
    """The ``*Details`` request object is the first positional argument of every client method."""
    if details_kwarg in kwargs:
        return kwargs[details_kwarg]
    return args[0] if args else None


def _metric_attributes(operation, request_model, response_model=None):
    attributes = {
        **Config.get_common_metrics_attributes(),
        GenAIAttributes.GEN_AI_PROVIDER_NAME: OCI_GENAI_PROVIDER,
        GenAIAttributes.GEN_AI_OPERATION_NAME: operation,
        GenAIAttributes.GEN_AI_REQUEST_MODEL: request_model,
    }
    if response_model:
        attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] = response_model
    return attributes


def _record_exception(span, error):
    span.set_attribute(ERROR_TYPE, error.__class__.__name__)
    span.record_exception(error)
    span.set_status(Status(StatusCode.ERROR, str(error)))


def is_streaming_response(operation, details, response):
    return is_stream_request(operation, details) and hasattr(getattr(response, "data", None), "events")


@dont_throw
def _handle_request(span, operation, details, instance, event_logger):
    set_request_attributes(span, operation, details, instance)
    if should_emit_events() and event_logger:
        emit_input_events(operation, details, event_logger)
    else:
        set_input_attributes(span, operation, details)


@dont_throw
def _handle_response(
    span, operation, response, request_model, token_histogram, duration_histogram, start_time, event_logger
):
    usage = set_response_attributes(span, operation, response)
    if should_emit_events() and event_logger:
        emit_response_events(operation, response, event_logger)
    else:
        set_output_attributes(span, operation, response)

    response_model = getattr(getattr(response, "data", None), "model_id", None)
    attributes = _metric_attributes(operation, request_model, response_model)
    if duration_histogram:
        duration_histogram.record(time.time() - start_time, attributes=attributes)
    record_usage_metrics(token_histogram, usage, attributes)


def _finish_streaming_span(
    span,
    operation,
    request_model,
    token_histogram,
    duration_histogram,
    event_logger,
    start_time,
    accumulator,
    error,
):
    """Callback invoked by ``OCIGenAIStreamWrapper`` once the SSE stream is exhausted (or fails)."""
    try:
        if error is not None:
            _record_exception(span, error)
        else:
            usage = set_streaming_response_attributes(span, accumulator)
            if should_emit_events() and event_logger:
                emit_streaming_response_events(accumulator, event_logger)
            else:
                set_streaming_output_attributes(span, accumulator)

            attributes = _metric_attributes(operation, request_model)
            if duration_histogram:
                duration_histogram.record(time.time() - start_time, attributes=attributes)
            record_usage_metrics(token_histogram, usage, attributes)

            if span.is_recording():
                span.set_status(Status(StatusCode.OK))
    except Exception as e:
        logger.debug("OpenLLMetry failed to finish OCI GenAI streaming span, error: %s", e)
        if Config.exception_logger:
            Config.exception_logger(e)
    finally:
        span.end()


def _with_tracer_wrapper(func):
    """Helper for providing tracer and metric instruments to wrapper functions."""

    def _with_tracer(tracer, token_histogram, duration_histogram, event_logger, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(
                tracer,
                token_histogram,
                duration_histogram,
                event_logger,
                to_wrap,
                wrapped,
                instance,
                args,
                kwargs,
            )

        return wrapper

    return _with_tracer


@_with_tracer_wrapper
def _wrap(
    tracer,
    token_histogram,
    duration_histogram,
    event_logger,
    to_wrap,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Instruments and calls every function defined in WRAPPED_METHODS."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    operation = to_wrap.get("operation")
    details = _resolve_details(args, kwargs, to_wrap.get("details_kwarg"))
    request_model = get_request_model(details)

    span = tracer.start_span(
        _span_name(operation, request_model),
        kind=SpanKind.CLIENT,
        attributes={
            GenAIAttributes.GEN_AI_PROVIDER_NAME: OCI_GENAI_PROVIDER,
            GenAIAttributes.GEN_AI_OPERATION_NAME: operation,
            GenAIAttributes.GEN_AI_REQUEST_MODEL: request_model,
        },
    )

    _handle_request(span, operation, details, instance, event_logger)

    start_time = time.time()
    try:
        response = wrapped(*args, **kwargs)
    except Exception as e:
        _record_exception(span, e)
        if duration_histogram:
            duration_histogram.record(
                time.time() - start_time,
                attributes={**_metric_attributes(operation, request_model), ERROR_TYPE: e.__class__.__name__},
            )
        span.end()
        raise

    if is_streaming_response(operation, details, response):
        # The span is completed once the caller has consumed ``response.data.events()``.
        response.data = OCIGenAIStreamWrapper(
            response.data,
            StreamAccumulator(),
            partial(
                _finish_streaming_span,
                span,
                operation,
                request_model,
                token_histogram,
                duration_histogram,
                event_logger,
                start_time,
            ),
        )
        return response

    _handle_response(
        span, operation, response, request_model, token_histogram, duration_histogram, start_time, event_logger
    )
    if span.is_recording():
        span.set_status(Status(StatusCode.OK))
    span.end()
    return response


class OCIGenAIInstrumentor(BaseInstrumentor):
    """An instrumentor for the OCI Python SDK's Generative AI inference client."""

    def __init__(
        self,
        exception_logger=None,
        use_attributes: Optional[bool] = None,
        get_common_metrics_attributes: Callable[[], dict] = lambda: {},
        use_legacy_attributes: Optional[bool] = None,
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

        # meter and histograms are inited here
        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        if is_metrics_enabled():
            token_histogram, duration_histogram = _create_metrics(meter)
        else:
            token_histogram, duration_histogram = None, None

        event_logger = None
        if not Config.use_legacy_attributes:
            logger_provider = kwargs.get("logger_provider")
            event_logger = get_logger(__name__, __version__, logger_provider=logger_provider)

        for wrapped_method in WRAPPED_METHODS:
            try:
                wrap_function_wrapper(
                    _CLIENT_MODULE,
                    f"{_CLIENT_CLASS}.{wrapped_method.get('method')}",
                    _wrap(tracer, token_histogram, duration_histogram, event_logger, wrapped_method),
                )
            except (AttributeError, ModuleNotFoundError):
                pass  # older SDK releases don't expose every operation (e.g. rerank_text)

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            try:
                unwrap(f"{_CLIENT_MODULE}.{_CLIENT_CLASS}", wrapped_method.get("method"))
            except (AttributeError, ModuleNotFoundError):
                pass
