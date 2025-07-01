"""OpenTelemetry SageMaker instrumentation"""

from functools import wraps
from typing import Collection

from opentelemetry import context as context_api
from opentelemetry._events import get_event_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.sagemaker.config import Config
from opentelemetry.instrumentation.sagemaker.event_emitter import (
    emit_choice_events,
    emit_message_event,
)
from opentelemetry.instrumentation.sagemaker.reusable_streaming_body import (
    ReusableStreamingBody,
)
from opentelemetry.instrumentation.sagemaker.span_utils import (
    set_call_request_attributes,
    set_call_response_attributes,
    set_call_span_attributes,
    set_stream_response_attributes,
)
from opentelemetry.instrumentation.sagemaker.streaming_wrapper import StreamingWrapper
from opentelemetry.instrumentation.sagemaker.utils import dont_throw, should_emit_events
from opentelemetry.instrumentation.sagemaker.version import __version__
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
)
from opentelemetry.trace import SpanKind, get_tracer
from wrapt import wrap_function_wrapper

_instruments = ("boto3 >= 1.28.57",)

WRAPPED_METHODS = [
    {
        "package": "botocore.client",
        "object": "ClientCreator",
        "method": "create_client",
    },
    {"package": "botocore.session", "object": "Session", "method": "create_client"},
]


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, event_logger, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, event_logger, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


@_with_tracer_wrapper
def _wrap(
    tracer,
    event_logger,
    to_wrap,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    if kwargs.get("service_name") == "sagemaker-runtime":
        client = wrapped(*args, **kwargs)
        client.invoke_endpoint = _instrumented_endpoint_invoke(
            client.invoke_endpoint, tracer, event_logger
        )
        client.invoke_endpoint_with_response_stream = (
            _instrumented_endpoint_invoke_with_response_stream(
                client.invoke_endpoint_with_response_stream, tracer, event_logger
            )
        )

        return client

    return wrapped(*args, **kwargs)


def _instrumented_endpoint_invoke(fn, tracer, event_logger):
    @wraps(fn)
    def with_instrumentation(*args, **kwargs):
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return fn(*args, **kwargs)

        with tracer.start_as_current_span(
            "sagemaker.completion", kind=SpanKind.CLIENT
        ) as span:
            response = fn(*args, **kwargs)
            _handle_call(span, event_logger, kwargs, response)

            return response

    return with_instrumentation


def _instrumented_endpoint_invoke_with_response_stream(fn, tracer, event_logger):
    @wraps(fn)
    def with_instrumentation(*args, **kwargs):
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return fn(*args, **kwargs)

        span = tracer.start_span("sagemaker.completion", kind=SpanKind.CLIENT)

        response = fn(*args, **kwargs)

        if span.is_recording():
            _handle_stream_call(span, event_logger, kwargs, response)

        return response

    return with_instrumentation


def _handle_stream_call(span, event_logger, kwargs, response):
    @dont_throw
    def stream_done(response_body):
        # Handle Request
        if should_emit_events() and event_logger is not None:
            emit_message_event(kwargs, event_logger)
        else:
            set_call_request_attributes(span, kwargs)

        set_call_span_attributes(span, kwargs, response)

        # Handle Response
        if should_emit_events() and event_logger is not None:
            emit_choice_events(response, event_logger)
        else:
            set_stream_response_attributes(span, response_body)

        span.end()

    response["Body"] = StreamingWrapper(response["Body"], stream_done)


@dont_throw
def _handle_call(span, event_logger, kwargs, response):
    # Handle Request
    if should_emit_events() and event_logger is not None:
        emit_message_event(kwargs, event_logger)
    else:
        set_call_request_attributes(span, kwargs)

    response["Body"] = ReusableStreamingBody(
        response["Body"]._raw_stream, response["Body"]._content_length
    )

    set_call_span_attributes(span, kwargs, response)

    # Handle Response
    if should_emit_events() and event_logger is not None:
        emit_choice_events(response, event_logger)
    else:
        raw_response = response["Body"].read()
        set_call_response_attributes(span, raw_response)


class SageMakerInstrumentor(BaseInstrumentor):
    """An instrumentor for Bedrock's client library."""

    def __init__(
        self,
        enrich_token_usage: bool = False,
        exception_logger=None,
        use_legacy_attributes: bool = True,
    ):
        super().__init__()
        Config.enrich_token_usage = enrich_token_usage
        Config.exception_logger = exception_logger
        Config.use_legacy_attributes = use_legacy_attributes

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

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
            wrap_function_wrapper(
                wrap_package,
                f"{wrap_object}.{wrap_method}",
                _wrap(tracer, event_logger, wrapped_method),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"{wrap_package}.{wrap_object}",
                wrapped_method.get("method"),
            )
