"""OpenTelemetry Bedrock instrumentation"""

from functools import wraps
import json
import logging
import os
import time
from typing import Collection
from opentelemetry.instrumentation.bedrock.config import Config
from opentelemetry.instrumentation.bedrock.events import (
    create_prompt_event,
    create_completion_event,
    create_tool_call_event,
)
from opentelemetry.instrumentation.bedrock.reusable_streaming_body import (
    ReusableStreamingBody,
)
from opentelemetry.instrumentation.bedrock.streaming_wrapper import StreamingWrapper
from opentelemetry.instrumentation.bedrock.utils import dont_throw
from opentelemetry.metrics import Counter, Histogram, Meter, get_meter
from wrapt import wrap_function_wrapper
import anthropic

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, SpanKind

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)

from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
    LLMRequestTypeValues,
    Meters,
)

from opentelemetry.instrumentation.bedrock.version import __version__


class MetricParams:
    def __init__(
        self,
        token_histogram: Histogram,
        choice_counter: Counter,
        duration_histogram: Histogram,
        exception_counter: Counter,
        event_logger=None,
        use_legacy_attributes: bool = True,
    ):
        self.vendor = ""
        self.model = ""
        self.is_stream = False
        self.token_histogram = token_histogram
        self.choice_counter = choice_counter
        self.duration_histogram = duration_histogram
        self.exception_counter = exception_counter
        self.event_logger = event_logger
        self.use_legacy_attributes = use_legacy_attributes
        self.start_time = time.time()


logger = logging.getLogger(__name__)

anthropic_client = anthropic.Anthropic()

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


def is_metrics_enabled() -> bool:
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(
        tracer,
        metric_params,
        to_wrap,
    ):
        def wrapper(wrapped, instance, args, kwargs):
            return func(
                tracer,
                metric_params,
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
                client.invoke_model, tracer, metric_params
            )
            client.invoke_model_with_response_stream = (
                _instrumented_model_invoke_with_response_stream(
                    client.invoke_model_with_response_stream, tracer, metric_params
                )
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


def _instrumented_model_invoke(fn, tracer, metric_params):
    @wraps(fn)
    def with_instrumentation(*args, **kwargs):
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return fn(*args, **kwargs)

        with tracer.start_as_current_span(
            "bedrock.completion", kind=SpanKind.CLIENT
        ) as span:
            response = fn(*args, **kwargs)

            if span.is_recording():
                _handle_call(span, kwargs, response, metric_params)

            return response

    return with_instrumentation


def _instrumented_model_invoke_with_response_stream(fn, tracer, metric_params):
    @wraps(fn)
    def with_instrumentation(*args, **kwargs):
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return fn(*args, **kwargs)

        span = tracer.start_span("bedrock.completion", kind=SpanKind.CLIENT)
        response = fn(*args, **kwargs)

        if span.is_recording():
            _handle_stream_call(span, kwargs, response, metric_params)

        return response

    return with_instrumentation


def _handle_stream_call(span, kwargs, response, metric_params):
    @dont_throw
    def stream_done(response_body):
        request_body = json.loads(kwargs.get("body"))

        (vendor, model) = kwargs.get("modelId").split(".")

        metric_params.vendor = vendor
        metric_params.model = model
        metric_params.is_stream = True

        if not metric_params.use_legacy_attributes:
            if metric_params.event_logger:
                # Log prompt event
                if vendor == "cohere":
                    _log_cohere_events(
                        metric_params.event_logger,
                        request_body,
                        response_body,
                        vendor,
                        model,
                    )
                elif vendor == "anthropic":
                    if "prompt" in request_body:
                        _log_anthropic_completion_events(
                            metric_params.event_logger,
                            request_body,
                            response_body,
                            vendor,
                            model,
                        )
                    elif "messages" in request_body:
                        _log_anthropic_messages_events(
                            metric_params.event_logger,
                            request_body,
                            response_body,
                            vendor,
                            model,
                        )
                elif vendor == "ai21":
                    _log_ai21_events(
                        metric_params.event_logger,
                        request_body,
                        response_body,
                        vendor,
                        model,
                    )
                elif vendor == "meta":
                    _log_llama_events(
                        metric_params.event_logger,
                        request_body,
                        response_body,
                        vendor,
                        model,
                    )
                elif vendor == "amazon":
                    _log_amazon_events(
                        metric_params.event_logger,
                        request_body,
                        response_body,
                        vendor,
                        model,
                    )
        else:
            _set_span_attribute(span, SpanAttributes.LLM_SYSTEM, vendor)
            _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, model)
            _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, model)

            if vendor == "cohere":
                _set_cohere_span_attributes(
                    span, request_body, response_body, metric_params
                )
            elif vendor == "anthropic":
                if "prompt" in request_body:
                    _set_anthropic_completion_span_attributes(
                        span, request_body, response_body, metric_params
                    )
                elif "messages" in request_body:
                    _set_anthropic_messages_span_attributes(
                        span, request_body, response_body, metric_params
                    )
            elif vendor == "ai21":
                _set_ai21_span_attributes(span, request_body, response_body, metric_params)
            elif vendor == "meta":
                _set_llama_span_attributes(span, request_body, response_body, metric_params)
            elif vendor == "amazon":
                _set_amazon_span_attributes(
                    span, request_body, response_body, metric_params
                )

        span.end()

    response["body"] = StreamingWrapper(response["body"], stream_done)


@dont_throw
def _handle_call(span, kwargs, response, metric_params):
    response["body"] = ReusableStreamingBody(
        response["body"]._raw_stream, response["body"]._content_length
    )
    request_body = json.loads(kwargs.get("body"))
    response_body = json.loads(response.get("body").read())

    (vendor, model) = kwargs.get("modelId").split(".")

    metric_params.vendor = vendor
    metric_params.model = model
    metric_params.is_stream = False

    _set_span_attribute(span, SpanAttributes.LLM_SYSTEM, vendor)
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, model)
    _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, model)

    if vendor == "cohere":
        _set_cohere_span_attributes(span, request_body, response_body, metric_params)
    elif vendor == "anthropic":
        if "prompt" in request_body:
            _set_anthropic_completion_span_attributes(
                span, request_body, response_body, metric_params
            )
        elif "messages" in request_body:
            _set_anthropic_messages_span_attributes(
                span, request_body, response_body, metric_params
            )
    elif vendor == "ai21":
        _set_ai21_span_attributes(span, request_body, response_body, metric_params)
    elif vendor == "meta":
        _set_llama_span_attributes(span, request_body, response_body, metric_params)
    elif vendor == "amazon":
        _set_amazon_span_attributes(span, request_body, response_body, metric_params)


def _record_usage_to_span(span, prompt_tokens, completion_tokens, metric_params):
    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
        prompt_tokens,
    )
    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
        completion_tokens,
    )
    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
        prompt_tokens + completion_tokens,
    )

    metric_attributes = _metric_shared_attributes(
        metric_params.vendor, metric_params.model, metric_params.is_stream
    )

    if metric_params.duration_histogram:
        duration = time.time() - metric_params.start_time
        metric_params.duration_histogram.record(
            duration,
            attributes=metric_attributes,
        )

    if (
        metric_params.token_histogram
        and type(prompt_tokens) is int
        and prompt_tokens >= 0
    ):
        metric_params.token_histogram.record(
            prompt_tokens,
            attributes={
                **metric_attributes,
                SpanAttributes.LLM_TOKEN_TYPE: "input",
            },
        )
    if (
        metric_params.token_histogram
        and type(completion_tokens) is int
        and completion_tokens >= 0
    ):
        metric_params.token_histogram.record(
            completion_tokens,
            attributes={
                **metric_attributes,
                SpanAttributes.LLM_TOKEN_TYPE: "output",
            },
        )


def _metric_shared_attributes(
    response_vendor: str, response_model: str, is_streaming: bool = False
):
    return {
        "vendor": response_vendor,
        SpanAttributes.LLM_RESPONSE_MODEL: response_model,
        SpanAttributes.LLM_SYSTEM: "bedrock",
        "stream": is_streaming,
    }


def _set_cohere_span_attributes(span, request_body, response_body, metric_params):
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.COMPLETION.value
    )
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TOP_P, request_body.get("p"))
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TEMPERATURE, request_body.get("temperature")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, request_body.get("max_tokens")
    )

    # based on contract at
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command-r-plus.html
    input_tokens = response_body.get("token_count", {}).get("prompt_tokens")
    output_tokens = response_body.get("token_count", {}).get("response_tokens")

    print("response_body", response_body)

    if input_tokens is None or output_tokens is None:
        meta = response_body.get("meta", {})
        billed_units = meta.get("billed_units", {})
        input_tokens = input_tokens or billed_units.get("input_tokens")
        output_tokens = output_tokens or billed_units.get("output_tokens")

    if input_tokens is not None and output_tokens is not None:
        _record_usage_to_span(
            span,
            input_tokens,
            output_tokens,
            metric_params,
        )

    if should_send_prompts():
        _set_span_attribute(
            span, f"{SpanAttributes.LLM_PROMPTS}.0.user", request_body.get("prompt")
        )

        for i, generation in enumerate(response_body.get("generations")):
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content",
                generation.get("text"),
            )


def _set_anthropic_completion_span_attributes(
    span, request_body, response_body, metric_params
):
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.COMPLETION.value
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TOP_P, request_body.get("top_p")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TEMPERATURE, request_body.get("temperature")
    )
    _set_span_attribute(
        span,
        SpanAttributes.LLM_REQUEST_MAX_TOKENS,
        request_body.get("max_tokens_to_sample"),
    )

    if (
        response_body.get("usage") is not None
        and response_body.get("usage").get("input_tokens") is not None
        and response_body.get("usage").get("output_tokens") is not None
    ):
        _record_usage_to_span(
            span,
            response_body.get("usage").get("input_tokens"),
            response_body.get("usage").get("output_tokens"),
            metric_params,
        )
    elif response_body.get("invocation_metrics") is not None:
        _record_usage_to_span(
            span,
            response_body.get("invocation_metrics").get("inputTokenCount"),
            response_body.get("invocation_metrics").get("outputTokenCount"),
            metric_params,
        )
    elif Config.enrich_token_usage:
        _record_usage_to_span(
            span,
            _count_anthropic_tokens([request_body.get("prompt")]),
            _count_anthropic_tokens([response_body.get("completion")]),
            metric_params,
        )

    if should_send_prompts():
        _set_span_attribute(
            span, f"{SpanAttributes.LLM_PROMPTS}.0.user", request_body.get("prompt")
        )
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
            response_body.get("completion"),
        )


def _set_anthropic_messages_span_attributes(
    span, request_body, response_body, metric_params
):
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.CHAT.value
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TOP_P, request_body.get("top_p")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TEMPERATURE, request_body.get("temperature")
    )
    _set_span_attribute(
        span,
        SpanAttributes.LLM_REQUEST_MAX_TOKENS,
        request_body.get("max_tokens"),
    )

    prompt_tokens = 0
    completion_tokens = 0
    if (
        response_body.get("usage") is not None
        and response_body.get("usage").get("input_tokens") is not None
        and response_body.get("usage").get("output_tokens") is not None
    ):
        prompt_tokens = response_body.get("usage").get("input_tokens")
        completion_tokens = response_body.get("usage").get("output_tokens")
        _record_usage_to_span(span, prompt_tokens, completion_tokens, metric_params)
    elif response_body.get("invocation_metrics") is not None:
        prompt_tokens = response_body.get("invocation_metrics").get("inputTokenCount")
        completion_tokens = response_body.get("invocation_metrics").get(
            "outputTokenCount"
        )
        _record_usage_to_span(span, prompt_tokens, completion_tokens, metric_params)
    elif Config.enrich_token_usage:
        messages = [message.get("content") for message in request_body.get("messages")]

        raw_messages = []
        for message in messages:
            if isinstance(message, str):
                raw_messages.append(message)
            else:
                raw_messages.extend([content.get("text") for content in message])
        prompt_tokens = _count_anthropic_tokens(raw_messages)
        completion_tokens = _count_anthropic_tokens(
            [content.get("text") for content in response_body.get("content")]
        )
        _record_usage_to_span(span, prompt_tokens, completion_tokens, metric_params)

    if should_send_prompts():
        for idx, message in enumerate(request_body.get("messages")):
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_PROMPTS}.{idx}.role", message.get("role")
            )
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.0.content",
                json.dumps(message.get("content")),
            )

        _set_span_attribute(
            span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", "assistant"
        )
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
            json.dumps(response_body.get("content")),
        )


def _count_anthropic_tokens(messages: list[str]):
    count = 0
    for message in messages:
        count += anthropic_client.count_tokens(text=message)
    return count


def _set_ai21_span_attributes(span, request_body, response_body, metric_params):
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.COMPLETION.value
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TOP_P, request_body.get("topP")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TEMPERATURE, request_body.get("temperature")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, request_body.get("maxTokens")
    )

    _record_usage_to_span(
        span,
        len(response_body.get("prompt").get("tokens")),
        len(response_body.get("completions")[0].get("data").get("tokens")),
        metric_params,
    )

    if should_send_prompts():
        _set_span_attribute(
            span, f"{SpanAttributes.LLM_PROMPTS}.0.user", request_body.get("prompt")
        )

        for i, completion in enumerate(response_body.get("completions")):
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content",
                completion.get("data").get("text"),
            )


def _set_llama_span_attributes(span, request_body, response_body, metric_params):
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.COMPLETION.value
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TOP_P, request_body.get("top_p")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TEMPERATURE, request_body.get("temperature")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, request_body.get("max_gen_len")
    )

    _record_usage_to_span(
        span,
        response_body.get("prompt_token_count"),
        response_body.get("generation_token_count"),
        metric_params,
    )

    if should_send_prompts():
        _set_span_attribute(
            span, f"{SpanAttributes.LLM_PROMPTS}.0.content", request_body.get("prompt")
        )
        _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")

        if response_body.get("generation"):
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant"
            )
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
                response_body.get("generation"),
            )
        else:
            for i, generation in enumerate(response_body.get("generations")):
                _set_span_attribute(
                    span, f"{SpanAttributes.LLM_COMPLETIONS}.{i}.role", "assistant"
                )
                _set_span_attribute(
                    span, f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content", generation
                )


def _set_amazon_span_attributes(span, request_body, response_body, metric_params):
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.COMPLETION.value
    )
    config = request_body.get("textGenerationConfig", {})
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TOP_P, config.get("topP"))
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TEMPERATURE, config.get("temperature")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, config.get("maxTokenCount")
    )

    _record_usage_to_span(
        span,
        response_body.get("inputTextTokenCount"),
        sum(int(result.get("tokenCount")) for result in response_body.get("results")),
        metric_params,
    )

    if should_send_prompts():
        _set_span_attribute(
            span, f"{SpanAttributes.LLM_PROMPTS}.0.user", request_body.get("inputText")
        )

        for i, result in enumerate(response_body.get("results")):
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content",
                result.get("outputText"),
            )


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

    return token_histogram, choice_counter, duration_histogram, exception_counter


def _log_cohere_events(event_logger, request_body, response_body, vendor, model):
    """Log events for Cohere model."""
    prompt = request_body.get("prompt", "")
    prompt_tokens = request_body.get("prompt_tokens", None)
    event_logger.add_event(
        create_prompt_event(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            vendor=vendor,
            model=model,
        )
    )

    completion = response_body.get("text", "")
    completion_tokens = response_body.get("completion_tokens", None)
    finish_reason = response_body.get("finish_reason", None)
    event_logger.add_event(
        create_completion_event(
            completion=completion,
            completion_tokens=completion_tokens,
            vendor=vendor,
            model=model,
            finish_reason=finish_reason,
        )
    )


def _log_anthropic_completion_events(event_logger, request_body, response_body, vendor, model):
    """Log events for Anthropic completion model."""
    prompt = request_body.get("prompt", "")
    prompt_tokens = None  # Anthropic doesn't provide token counts in request
    event_logger.add_event(
        create_prompt_event(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            vendor=vendor,
            model=model,
        )
    )

    completion = response_body.get("completion", "")
    completion_tokens = None  # Anthropic doesn't provide token counts in response
    finish_reason = response_body.get("stop_reason", None)
    event_logger.add_event(
        create_completion_event(
            completion=completion,
            completion_tokens=completion_tokens,
            vendor=vendor,
            model=model,
            finish_reason=finish_reason,
        )
    )


def _log_anthropic_messages_events(event_logger, request_body, response_body, vendor, model):
    """Log events for Anthropic messages model."""
    messages = request_body.get("messages", [])
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        event_logger.add_event(
            create_prompt_event(
                prompt=content,
                role=role,
                vendor=vendor,
                model=model,
            )
        )

    completion = response_body.get("content", [])
    if isinstance(completion, list):
        for content in completion:
            role = content.get("role", "")
            text = content.get("text", "")
            event_logger.add_event(
                create_completion_event(
                    completion=text,
                    role=role,
                    vendor=vendor,
                    model=model,
                )
            )
    else:
        event_logger.add_event(
            create_completion_event(
                completion=completion,
                vendor=vendor,
                model=model,
            )
        )


def _log_ai21_events(event_logger, request_body, response_body, vendor, model):
    """Log events for AI21 model."""
    prompt = request_body.get("prompt", "")
    prompt_tokens = None  # AI21 doesn't provide token counts in request
    event_logger.add_event(
        create_prompt_event(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            vendor=vendor,
            model=model,
        )
    )

    completion = response_body.get("completions", [{}])[0].get("data", {}).get("text", "")
    completion_tokens = None  # AI21 doesn't provide token counts in response
    finish_reason = response_body.get("completions", [{}])[0].get("finishReason", {}).get("reason", None)
    event_logger.add_event(
        create_completion_event(
            completion=completion,
            completion_tokens=completion_tokens,
            vendor=vendor,
            model=model,
            finish_reason=finish_reason,
        )
    )


def _log_llama_events(event_logger, request_body, response_body, vendor, model):
    """Log events for Llama model."""
    prompt = request_body.get("prompt", "")
    prompt_tokens = None  # Llama doesn't provide token counts in request
    event_logger.add_event(
        create_prompt_event(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            vendor=vendor,
            model=model,
        )
    )

    completion = response_body.get("generation", "")
    completion_tokens = None  # Llama doesn't provide token counts in response
    finish_reason = response_body.get("stop_reason", None)
    event_logger.add_event(
        create_completion_event(
            completion=completion,
            completion_tokens=completion_tokens,
            vendor=vendor,
            model=model,
            finish_reason=finish_reason,
        )
    )


def _log_amazon_events(event_logger, request_body, response_body, vendor, model):
    """Log events for Amazon model."""
    prompt = request_body.get("inputText", "")
    prompt_tokens = None  # Amazon doesn't provide token counts in request
    event_logger.add_event(
        create_prompt_event(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            vendor=vendor,
            model=model,
        )
    )

    completion = response_body.get("results", [{}])[0].get("outputText", "")
    completion_tokens = None  # Amazon doesn't provide token counts in response
    finish_reason = response_body.get("results", [{}])[0].get("completionReason", None)
    event_logger.add_event(
        create_completion_event(
            completion=completion,
            completion_tokens=completion_tokens,
            vendor=vendor,
            model=model,
            finish_reason=finish_reason,
        )
    )


class BedrockInstrumentor(BaseInstrumentor):
    """An instrumentor for Bedrock's client library."""

    def __init__(self, enrich_token_usage: bool = False, exception_logger=None, use_legacy_attributes: bool = True):
        super().__init__()
        self.enrich_token_usage = enrich_token_usage
        self.exception_logger = exception_logger
        self.use_legacy_attributes = use_legacy_attributes

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        event_logger = kwargs.get("event_logger")

        token_histogram, choice_counter, duration_histogram, exception_counter = _create_metrics(meter)

        metric_params = MetricParams(
            token_histogram=token_histogram,
            choice_counter=choice_counter,
            duration_histogram=duration_histogram,
            exception_counter=exception_counter,
            event_logger=event_logger,
            use_legacy_attributes=self.use_legacy_attributes,
        )

        for wrapped_method in WRAPPED_METHODS:
            wrap_function_wrapper(
                wrapped_method["package"],
                wrapped_method["object"] + "." + wrapped_method["method"],
                _wrap(tracer, metric_params, wrapped_method),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"{wrap_package}.{wrap_object}",
                wrapped_method.get("method"),
            )
