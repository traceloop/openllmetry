"""OpenTelemetry Bedrock instrumentation"""

import json
import logging
import os
import time
from functools import partial, wraps
from typing import Collection

from opentelemetry import context as context_api
from opentelemetry._events import get_event_logger
from opentelemetry.instrumentation.bedrock.config import Config
from opentelemetry.instrumentation.bedrock.event_emitter import (
    emit_choice_events,
    emit_input_events_converse,
    emit_message_events,
    emit_response_event_converse,
    emit_streaming_converse_response_event,
    emit_streaming_response_event,
)
from opentelemetry.instrumentation.bedrock.guardrail import (
    guardrail_converse,
    guardrail_handling,
)
from opentelemetry.instrumentation.bedrock.prompt_caching import prompt_caching_handling
from opentelemetry.instrumentation.bedrock.reusable_streaming_body import (
    ReusableStreamingBody,
)
from opentelemetry.instrumentation.bedrock.span_utils import (
    converse_usage_record,
    set_converse_input_prompt_span_attributes,
    set_converse_model_span_attributes,
    set_converse_response_span_attributes,
    set_converse_streaming_response_span_attributes,
    set_model_choice_span_attributes,
    set_model_message_span_attributes,
    set_model_span_attributes,
)
from opentelemetry.instrumentation.bedrock.streaming_wrapper import StreamingWrapper
from opentelemetry.instrumentation.bedrock.utils import (
    dont_throw,
    should_emit_events,
)
from opentelemetry.instrumentation.bedrock.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.metrics import Counter, Histogram, Meter, get_meter
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    Meters,
)
from opentelemetry.trace import Span, SpanKind, get_tracer
from wrapt import wrap_function_wrapper


class MetricParams:
    def __init__(
        self,
        token_histogram: Histogram,
        choice_counter: Counter,
        duration_histogram: Histogram,
        exception_counter: Counter,
        guardrail_activation: Counter,
        guardrail_latency_histogram: Histogram,
        guardrail_coverage: Counter,
        guardrail_sensitive_info: Counter,
        guardrail_topic: Counter,
        guardrail_content: Counter,
        guardrail_words: Counter,
        prompt_caching: Counter,
    ):
        self.vendor = ""
        self.model = ""
        self.is_stream = False
        self.token_histogram = token_histogram
        self.choice_counter = choice_counter
        self.duration_histogram = duration_histogram
        self.exception_counter = exception_counter
        self.guardrail_activation = guardrail_activation
        self.guardrail_latency_histogram = guardrail_latency_histogram
        self.guardrail_coverage = guardrail_coverage
        self.guardrail_sensitive_info = guardrail_sensitive_info
        self.guardrail_topic = guardrail_topic
        self.guardrail_content = guardrail_content
        self.guardrail_words = guardrail_words
        self.prompt_caching = prompt_caching
        self.start_time = time.time()


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

_BEDROCK_INVOKE_SPAN_NAME = "bedrock.completion"
_BEDROCK_CONVERSE_SPAN_NAME = "bedrock.converse"


def is_metrics_enabled() -> bool:
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(
        tracer,
        metric_params,
        event_logger,
        to_wrap,
    ):
        def wrapper(wrapped, instance, args, kwargs):
            return func(
                tracer,
                metric_params,
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
    metric_params,
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

    if kwargs.get("service_name") == "bedrock-runtime":
        try:
            start_time = time.time()
            metric_params.start_time = time.time()
            client = wrapped(*args, **kwargs)
            client.invoke_model = _instrumented_model_invoke(
                client.invoke_model, tracer, metric_params, event_logger
            )
            client.invoke_model_with_response_stream = (
                _instrumented_model_invoke_with_response_stream(
                    client.invoke_model_with_response_stream,
                    tracer,
                    metric_params,
                    event_logger,
                )
            )
            client.converse = _instrumented_converse(
                client.converse, tracer, metric_params, event_logger
            )
            client.converse_stream = _instrumented_converse_stream(
                client.converse_stream, tracer, metric_params, event_logger
            )
            return client
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time if "start_time" in locals() else 0

            attributes = {
                "error.type": e.__class__.__name__,
            }

            if duration > 0 and metric_params.duration_histogram:
                metric_params.duration_histogram.record(duration, attributes=attributes)
            if metric_params.exception_counter:
                metric_params.exception_counter.add(1, attributes=attributes)

            raise e

    return wrapped(*args, **kwargs)


def _instrumented_model_invoke(fn, tracer, metric_params, event_logger):
    @wraps(fn)
    def with_instrumentation(*args, **kwargs):
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return fn(*args, **kwargs)

        with tracer.start_as_current_span(
            _BEDROCK_INVOKE_SPAN_NAME, kind=SpanKind.CLIENT
        ) as span:
            response = fn(*args, **kwargs)
            _handle_call(span, kwargs, response, metric_params, event_logger)
            return response

    return with_instrumentation


def _instrumented_model_invoke_with_response_stream(
    fn, tracer, metric_params, event_logger
):
    @wraps(fn)
    def with_instrumentation(*args, **kwargs):
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return fn(*args, **kwargs)

        span = tracer.start_span(_BEDROCK_INVOKE_SPAN_NAME, kind=SpanKind.CLIENT)

        response = fn(*args, **kwargs)
        _handle_stream_call(span, kwargs, response, metric_params, event_logger)

        return response

    return with_instrumentation


def _instrumented_converse(fn, tracer, metric_params, event_logger):
    # see
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html
    # for the request/response format
    @wraps(fn)
    def with_instrumentation(*args, **kwargs):
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return fn(*args, **kwargs)

        with tracer.start_as_current_span(
            _BEDROCK_CONVERSE_SPAN_NAME, kind=SpanKind.CLIENT
        ) as span:
            response = fn(*args, **kwargs)
            _handle_converse(span, kwargs, response, metric_params, event_logger)

            return response

    return with_instrumentation


def _instrumented_converse_stream(fn, tracer, metric_params, event_logger):
    @wraps(fn)
    def with_instrumentation(*args, **kwargs):
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return fn(*args, **kwargs)

        span = tracer.start_span(_BEDROCK_CONVERSE_SPAN_NAME, kind=SpanKind.CLIENT)
        response = fn(*args, **kwargs)
        if span.is_recording():
            _handle_converse_stream(span, kwargs, response, metric_params, event_logger)

        return response

    return with_instrumentation


@dont_throw
def _handle_stream_call(span, kwargs, response, metric_params, event_logger):

    (provider, model_vendor, model) = _get_vendor_model(kwargs.get("modelId"))
    request_body = json.loads(kwargs.get("body"))

    headers = {}
    if "ResponseMetadata" in response:
        headers = response.get("ResponseMetadata").get("HTTPHeaders", {})

    @dont_throw
    def stream_done(response_body):

        metric_params.vendor = provider
        metric_params.model = model
        metric_params.is_stream = True

        prompt_caching_handling(headers, provider, model, metric_params)
        guardrail_handling(span, response_body, provider, model, metric_params)

        if span.is_recording():
            set_model_span_attributes(
                provider,
                model_vendor,
                model,
                span,
                request_body,
                response_body,
                headers,
                metric_params,
                kwargs,
            )
        if should_emit_events() and event_logger:
            emit_message_events(event_logger, kwargs)
            emit_streaming_response_event(response_body, event_logger)
        else:
            set_model_message_span_attributes(model_vendor, span, request_body)
            set_model_choice_span_attributes(model_vendor, span, response_body)

        span.end()

    response["body"] = StreamingWrapper(
        response["body"], stream_done_callback=stream_done
    )


@dont_throw
def _handle_call(span: Span, kwargs, response, metric_params, event_logger):
    response["body"] = ReusableStreamingBody(
        response["body"]._raw_stream, response["body"]._content_length
    )
    request_body = json.loads(kwargs.get("body"))
    response_body = json.loads(response.get("body").read())
    headers = {}
    if "ResponseMetadata" in response:
        headers = response.get("ResponseMetadata").get("HTTPHeaders", {})

    (provider, model_vendor, model) = _get_vendor_model(kwargs.get("modelId"))
    metric_params.vendor = provider
    metric_params.model = model
    metric_params.is_stream = False

    prompt_caching_handling(headers, provider, model, metric_params)
    guardrail_handling(span, response_body, provider, model, metric_params)

    if span.is_recording():
        set_model_span_attributes(
            provider,
            model_vendor,
            model,
            span,
            request_body,
            response_body,
            headers,
            metric_params,
            kwargs,
        )

    if should_emit_events() and event_logger:
        emit_message_events(event_logger, kwargs)
        emit_choice_events(event_logger, response)
    else:
        set_model_message_span_attributes(model_vendor, span, request_body)
        set_model_choice_span_attributes(model_vendor, span, response_body)


@dont_throw
def _handle_converse(span, kwargs, response, metric_params, event_logger):
    (provider, model_vendor, model) = _get_vendor_model(kwargs.get("modelId"))
    guardrail_converse(span, response, provider, model, metric_params)

    set_converse_model_span_attributes(span, provider, model, kwargs)

    converse_usage_record(span, response, metric_params)

    if should_emit_events() and event_logger:
        emit_input_events_converse(kwargs, event_logger)
        emit_response_event_converse(response, event_logger)
    else:
        set_converse_input_prompt_span_attributes(kwargs, span)
        set_converse_response_span_attributes(response, span)


@dont_throw
def _handle_converse_stream(span, kwargs, response, metric_params, event_logger):
    (provider, model_vendor, model) = _get_vendor_model(kwargs.get("modelId"))

    set_converse_model_span_attributes(span, provider, model, kwargs)

    if should_emit_events() and event_logger:
        emit_input_events_converse(kwargs, event_logger)
    else:
        set_converse_input_prompt_span_attributes(kwargs, span)

    stream = response.get("stream")
    role = "unknown"
    if stream:

        def handler(func):
            def wrap(*args, **kwargs):
                response_msg = kwargs.pop("response_msg")
                span = kwargs.pop("span")
                event = func(*args, **kwargs)
                nonlocal role
                if "contentBlockDelta" in event:
                    response_msg.append(event["contentBlockDelta"]["delta"]["text"])
                elif "messageStart" in event:
                    role = event["messageStart"]["role"]
                elif "metadata" in event:
                    # last message sent
                    guardrail_converse(span, event["metadata"], provider, model, metric_params)
                    converse_usage_record(span, event["metadata"], metric_params)
                    span.end()
                elif "messageStop" in event:
                    if should_emit_events() and event_logger:
                        emit_streaming_converse_response_event(
                            event_logger,
                            response_msg,
                            role,
                            event.get("messageStop", {}).get("stopReason", "unknown"),
                        )
                    else:
                        set_converse_streaming_response_span_attributes(
                            response_msg, role, span
                        )

                return event

            return partial(wrap, response_msg=[], span=span)

        stream._parse_event = handler(stream._parse_event)


def _get_vendor_model(modelId):
    # Docs:
    # https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html#inference-profiles-support-system
    provider = "AWS"
    model_vendor = "imported_model"
    model = modelId

    if modelId is not None and modelId.startswith("arn"):
        components = modelId.split(":")
        if len(components) > 5:
            inf_profile = components[5].split("/")
            if len(inf_profile) == 2:
                if "." in inf_profile[1]:
                    (model_vendor, model) = _cross_region_check(inf_profile[1])
    elif modelId is not None and "." in modelId:
        (model_vendor, model) = _cross_region_check(modelId)

    return provider, model_vendor, model


def _cross_region_check(value):
    prefixes = ["us", "us-gov", "eu", "apac"]
    if any(value.startswith(prefix + ".") for prefix in prefixes):
        parts = value.split(".")
        if len(parts) > 2:
            parts.pop(0)
        return parts[0], parts[1]
    else:
        (model_vendor, model) = value.split(".", 1)
    return model_vendor, model


class GuardrailMeters:
    LLM_BEDROCK_GUARDRAIL_ACTIVATION = "gen_ai.bedrock.guardrail.activation"
    LLM_BEDROCK_GUARDRAIL_LATENCY = "gen_ai.bedrock.guardrail.latency"
    LLM_BEDROCK_GUARDRAIL_COVERAGE = "gen_ai.bedrock.guardrail.coverage"
    LLM_BEDROCK_GUARDRAIL_SENSITIVE = "gen_ai.bedrock.guardrail.sensitive_info"
    LLM_BEDROCK_GUARDRAIL_TOPICS = "gen_ai.bedrock.guardrail.topics"
    LLM_BEDROCK_GUARDRAIL_CONTENT = "gen_ai.bedrock.guardrail.content"
    LLM_BEDROCK_GUARDRAIL_WORDS = "gen_ai.bedrock.guardrail.words"


class PromptCaching:
    # will be moved under the AI SemConv. Not namespaced since also OpenAI supports this.
    LLM_BEDROCK_PROMPT_CACHING = "gen_ai.prompt.caching"


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
        # TODO: will fix this in future as a consolidation for semantic convention
        name="llm.bedrock.completions.exceptions",
        unit="time",
        description="Number of exceptions occurred during chat completions",
    )

    # Guardrail metrics
    guardrail_activation = meter.create_counter(
        name=GuardrailMeters.LLM_BEDROCK_GUARDRAIL_ACTIVATION,
        unit="",
        description="Number of guardrail activation",
    )

    guardrail_latency_histogram = meter.create_histogram(
        name=GuardrailMeters.LLM_BEDROCK_GUARDRAIL_LATENCY,
        unit="ms",
        description="GenAI guardrail latency",
    )

    guardrail_coverage = meter.create_counter(
        name=GuardrailMeters.LLM_BEDROCK_GUARDRAIL_COVERAGE,
        unit="char",
        description="GenAI guardrail coverage",
    )

    guardrail_sensitive_info = meter.create_counter(
        name=GuardrailMeters.LLM_BEDROCK_GUARDRAIL_SENSITIVE,
        unit="",
        description="GenAI guardrail sensitive information protection",
    )

    guardrail_topic = meter.create_counter(
        name=GuardrailMeters.LLM_BEDROCK_GUARDRAIL_TOPICS,
        unit="",
        description="GenAI guardrail topics protection",
    )

    guardrail_content = meter.create_counter(
        name=GuardrailMeters.LLM_BEDROCK_GUARDRAIL_CONTENT,
        unit="",
        description="GenAI guardrail content filter protection",
    )

    guardrail_words = meter.create_counter(
        name=GuardrailMeters.LLM_BEDROCK_GUARDRAIL_WORDS,
        unit="",
        description="GenAI guardrail words filter protection",
    )

    # Prompt Caching
    prompt_caching = meter.create_counter(
        name=PromptCaching.LLM_BEDROCK_PROMPT_CACHING,
        unit="",
        description="Number of cached tokens",
    )

    return (
        token_histogram,
        choice_counter,
        duration_histogram,
        exception_counter,
        guardrail_activation,
        guardrail_latency_histogram,
        guardrail_coverage,
        guardrail_sensitive_info,
        guardrail_topic,
        guardrail_content,
        guardrail_words,
        prompt_caching,
    )


class BedrockInstrumentor(BaseInstrumentor):
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

        # meter and counters are inited here
        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        if is_metrics_enabled():
            (
                token_histogram,
                choice_counter,
                duration_histogram,
                exception_counter,
                guardrail_activation,
                guardrail_latency_histogram,
                guardrail_coverage,
                guardrail_sensitive_info,
                guardrail_topic,
                guardrail_content,
                guardrail_words,
                prompt_caching,
            ) = _create_metrics(meter)
        else:
            (
                token_histogram,
                choice_counter,
                duration_histogram,
                exception_counter,
                guardrail_activation,
                guardrail_latency_histogram,
                guardrail_coverage,
                guardrail_sensitive_info,
                guardrail_topic,
                guardrail_content,
                guardrail_words,
                prompt_caching,
            ) = (None, None, None, None, None, None, None, None, None, None, None, None)

        metric_params = MetricParams(
            token_histogram,
            choice_counter,
            duration_histogram,
            exception_counter,
            guardrail_activation,
            guardrail_latency_histogram,
            guardrail_coverage,
            guardrail_sensitive_info,
            guardrail_topic,
            guardrail_content,
            guardrail_words,
            prompt_caching,
        )

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
                _wrap(
                    tracer,
                    metric_params,
                    event_logger,
                    wrapped_method,
                ),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"{wrap_package}.{wrap_object}",
                wrapped_method.get("method"),
            )
