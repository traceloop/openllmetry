"""OpenTelemetry SageMaker instrumentation"""

import json
import logging
import os
from functools import wraps
from typing import Collection, Union

from opentelemetry import context as context_api
from opentelemetry._events import Event, EventLogger, get_event_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.sagemaker.config import Config
from opentelemetry.instrumentation.sagemaker.reusable_streaming_body import (
    ReusableStreamingBody,
)
from opentelemetry.instrumentation.sagemaker.streaming_wrapper import StreamingWrapper
from opentelemetry.instrumentation.sagemaker.utils import dont_throw, is_content_enabled
from opentelemetry.instrumentation.sagemaker.version import __version__
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import SpanKind, get_tracer
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("boto3 >= 1.28.57",)

WRAPPED_METHODS = [
    {
        "package": "botocore.client",
        "object": "ClientCreator",
        "method": "create_client",
    },
    {"package": "botocore.session", "object": "Session", "method": "create_client"},
]


def should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _emit_input_events(event_logger: EventLogger, kwargs):
    attributes = {GenAIAttributes.GEN_AI_SYSTEM: "sagemaker"}
    input_body = json.loads(kwargs.get("Body"))
    body = {"content": input_body.get("inputs", "")} if is_content_enabled() else {}
    event_logger.emit(
        Event(name="gen_ai.user.message", attributes=attributes, body=body)
    )


def _emit_response_events(event_logger: EventLogger, response: dict):
    attributes = {GenAIAttributes.GEN_AI_SYSTEM: "sagemaker"}
    response_body: Union[StreamingWrapper, ReusableStreamingBody, None] = response.get(
        "Body"
    )

    if isinstance(response_body, StreamingWrapper):
        body = {"index": 0, "finish_reason": "unknown", "message": {}}
        if is_content_enabled():
            body["message"]["content"] = response_body._accumulating_body
        event_logger.emit(Event(name="gen_ai.choice", attributes=attributes, body=body))

    elif isinstance(response_body, ReusableStreamingBody):
        for i, gen in enumerate(json.loads(response_body.read())):
            body = {"index": i, "finish_reason": "unknown", "message": {}}
            if is_content_enabled():
                body["message"]["content"] = gen.get("generated_text")
            event_logger.emit(
                Event(name="gen_ai.choice", attributes=attributes, body=body)
            )


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
    event_logger: Union[EventLogger, None],
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


def _instrumented_endpoint_invoke(fn, tracer, event_logger: Union[EventLogger, None]):
    @wraps(fn)
    def with_instrumentation(*args, **kwargs):
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return fn(*args, **kwargs)

        with tracer.start_as_current_span(
            "sagemaker.completion", kind=SpanKind.CLIENT
        ) as span:
            if not Config.use_legacy_attributes and event_logger is not None:
                _emit_input_events(event_logger, kwargs)

            response = fn(*args, **kwargs)

            if span.is_recording():
                _handle_call(span, kwargs, response)

            if not Config.use_legacy_attributes and event_logger is not None:
                _emit_response_events(event_logger, response)

            return response

    return with_instrumentation


def _instrumented_endpoint_invoke_with_response_stream(
    fn, tracer, event_logger: Union[EventLogger, None]
):
    @wraps(fn)
    def with_instrumentation(*args, **kwargs):
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return fn(*args, **kwargs)

        span = tracer.start_span("sagemaker.completion", kind=SpanKind.CLIENT)

        if not Config.use_legacy_attributes and event_logger is not None:
            _emit_input_events(event_logger, kwargs)

        response = fn(*args, **kwargs)

        if span.is_recording():
            _handle_stream_call(span, kwargs, response)

        if not Config.use_legacy_attributes and event_logger is not None:
            _emit_response_events(event_logger, response)

        return response

    return with_instrumentation


def _handle_stream_call(span, kwargs, response):
    @dont_throw
    def stream_done(response_body):
        request_body = json.loads(kwargs.get("Body"))

        endpoint_name = kwargs.get("EndpointName")

        _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, endpoint_name)
        _set_span_attribute(
            span, SpanAttributes.TRACELOOP_ENTITY_INPUT, json.dumps(request_body)
        )
        _set_span_attribute(
            span, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json.dumps(response_body)
        )

        span.end()

    response["Body"] = StreamingWrapper(response["Body"], stream_done)


@dont_throw
def _handle_call(span, kwargs, response):
    response["Body"] = ReusableStreamingBody(
        response["Body"]._raw_stream, response["Body"]._content_length
    )
    request_body = json.loads(kwargs.get("Body"))
    response_body = json.loads(response.get("Body").read())

    endpoint_name = kwargs.get("EndpointName")

    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, endpoint_name)
    _set_span_attribute(
        span, SpanAttributes.TRACELOOP_ENTITY_INPUT, json.dumps(request_body)
    )
    _set_span_attribute(
        span, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json.dumps(response_body)
    )


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

        if Config.use_legacy_attributes:
            event_logger = None
        else:
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
