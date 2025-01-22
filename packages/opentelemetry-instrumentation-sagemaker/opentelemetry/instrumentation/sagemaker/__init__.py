"""OpenTelemetry SageMaker instrumentation"""

from functools import wraps
import json
import logging
import os
from typing import Collection

from opentelemetry.instrumentation.sagemaker.config import Config
from opentelemetry.instrumentation.sagemaker.events import (
    prompt_to_event,
    completion_to_event,
)
from opentelemetry.instrumentation.sagemaker.reusable_streaming_body import (
    ReusableStreamingBody,
)
from opentelemetry.instrumentation.sagemaker.streaming_wrapper import StreamingWrapper
from opentelemetry.instrumentation.sagemaker.utils import dont_throw
from wrapt import wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry._events import EventLogger
from opentelemetry.trace import get_tracer, SpanKind
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)

from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.instrumentation.sagemaker.version import __version__

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


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, event_logger, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, event_logger, to_wrap, wrapped, instance, args, kwargs)
        return wrapper
    return _with_tracer


@_with_tracer_wrapper
def _wrap(tracer, event_logger, to_wrap, wrapped, instance, args, kwargs, config):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    if kwargs.get("service_name") == "sagemaker-runtime":
        client = wrapped(*args, **kwargs)
        client.invoke_endpoint = _instrumented_endpoint_invoke(
            client.invoke_endpoint, tracer, event_logger, config
        )
        client.invoke_endpoint_with_response_stream = (
            _instrumented_endpoint_invoke_with_response_stream(
                client.invoke_endpoint_with_response_stream, tracer, event_logger, config
            )
        )

        return client

    return wrapped(*args, **kwargs)


def _instrumented_endpoint_invoke(fn, tracer, event_logger, config):
    @wraps(fn)
    def with_instrumentation(*args, **kwargs):
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return fn(*args, **kwargs)

        endpoint_name = kwargs.get("EndpointName", "")
        body = kwargs.get("Body", "{}")
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                body = {"content": body}

        with tracer.start_as_current_span(
            "sagemaker.completion", kind=SpanKind.CLIENT
        ) as span:
            if span.is_recording():
                if config.use_legacy_attributes:
                    span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, endpoint_name)
                    if isinstance(body, dict):
                        for key, value in body.items():
                            span.set_attribute(
                                f"{SpanAttributes.LLM_PROMPTS}.{key}",
                                str(value) if value is not None else None,
                            )
                    else:
                        span.set_attribute(
                            f"{SpanAttributes.LLM_PROMPTS}.0.content",
                            str(body) if body is not None else None,
                        )
                else:
                    event_logger.emit(
                        prompt_to_event(body, endpoint_name, config.capture_content)
                    )

            try:
                response = fn(*args, **kwargs)

                if span.is_recording():
                    response_body = response.get("Body", "{}")
                    if isinstance(response_body, bytes):
                        response_body = response_body.decode("utf-8")
                    if isinstance(response_body, str):
                        try:
                            response_body = json.loads(response_body)
                        except json.JSONDecodeError:
                            response_body = {"content": response_body}

                    if config.use_legacy_attributes:
                        span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, endpoint_name)
                        if isinstance(response_body, dict):
                            for key, value in response_body.items():
                                span.set_attribute(
                                    f"{SpanAttributes.LLM_COMPLETIONS}.{key}",
                                    str(value) if value is not None else None,
                                )
                        else:
                            span.set_attribute(
                                f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
                                str(response_body) if response_body is not None else None,
                            )
                    else:
                        event_logger.emit(
                            completion_to_event(response_body, endpoint_name, config.capture_content)
                        )

                    if config.enrich_token_usage:
                        try:
                            if isinstance(response_body, dict) and "usage" in response_body:
                                usage = response_body["usage"]
                                if "prompt_tokens" in usage:
                                    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, usage["prompt_tokens"])
                                if "completion_tokens" in usage:
                                    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, usage["completion_tokens"])
                                if "total_tokens" in usage:
                                    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, usage["total_tokens"])
                        except Exception as e:
                            if config.exception_logger:
                                config.exception_logger(e)
                            logger.warning("Failed to enrich token usage: %s", str(e))

                return response

            except Exception as ex:
                if span.is_recording():
                    span.set_status(Status(StatusCode.ERROR))
                    span.record_exception(ex)
                raise

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

def _instrumented_endpoint_invoke_with_response_stream(fn, tracer, event_logger, config):
    @wraps(fn)
    def with_instrumentation(*args, **kwargs):
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return fn(*args, **kwargs)

        endpoint_name = kwargs.get("EndpointName", "")
        body = kwargs.get("Body", "{}")
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                body = {"content": body}

        span = tracer.start_span("sagemaker.completion", kind=SpanKind.CLIENT)
        if span.is_recording():
            if config.use_legacy_attributes:
                span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, endpoint_name)
                if isinstance(body, dict):
                    for key, value in body.items():
                        span.set_attribute(
                            f"{SpanAttributes.LLM_PROMPTS}.{key}",
                            str(value) if value is not None else None,
                        )
                else:
                    span.set_attribute(
                        f"{SpanAttributes.LLM_PROMPTS}.0.content",
                        str(body) if body is not None else None,
                    )
            else:
                event_logger.emit(
                    prompt_to_event(body, endpoint_name, config.capture_content)
                )

        try:
            response = fn(*args, **kwargs)

            if span.is_recording():
                if config.use_legacy_attributes:
                    span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, endpoint_name)
                else:
                    event_logger.emit(
                        completion_to_event({"streaming": True}, endpoint_name, config.capture_content)
                    )

            return response

        except Exception as ex:
            if span.is_recording():
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(ex)
            raise

    return with_instrumentation


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


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


def should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


class SageMakerInstrumentor(BaseInstrumentor):
    """An instrumentor for Bedrock's client library."""

    def __init__(self, enrich_token_usage: bool = False, exception_logger=None):
        self._exception_logger = exception_logger
        self._enrich_token_usage = enrich_token_usage
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        self._config = Config(
            use_legacy_attributes=kwargs.get("use_legacy_attributes", True),
            capture_content=kwargs.get("capture_content", True),
            exception_logger=self._exception_logger,
            enrich_token_usage=self._enrich_token_usage,
        )

        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        event_logger = EventLogger(__name__, tracer_provider)

        for wrapped_method in WRAPPED_METHODS:
            wrap_function_wrapper(
                wrapped_method["package"],
                f"{wrapped_method['object']}.{wrapped_method['method']}",
                _wrap(tracer, event_logger, wrapped_method, self._config),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            unwrap(
                f"{wrapped_method['package']}.{wrapped_method['object']}",
                wrapped_method["method"],
            )
