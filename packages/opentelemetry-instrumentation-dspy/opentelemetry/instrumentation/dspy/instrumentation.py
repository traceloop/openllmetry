import logging
import os
import time
from typing import Collection

from wrapt import wrap_function_wrapper
from opentelemetry.trace import SpanKind, Tracer, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.metrics import Histogram, Meter, get_meter
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.dspy.version import __version__
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiOperationNameValues,
    GenAiSystemValues,
)
from opentelemetry.semconv_ai import GenAISystem, Meters, SpanAttributes, TraceloopSpanKindValues

from .utils import (
    get_token_usage,
    messages_to_otel_input,
    response_to_otel_output,
    set_span_attribute,
)

logger = logging.getLogger(__name__)

_instruments = ("dspy >= 2.5.0",)

_DSPY_SIGNATURE = "dspy.signature"
_DSPY_CACHE_HIT = "dspy.cache_hit"

# Maps LiteLLM vendor prefixes (e.g. "openai" in "openai/gpt-4") to OTel provider name values.
# Uses GenAISystem (semconv-ai) and GenAiSystemValues (OTel upstream) — no raw strings.
_LITELLM_PREFIX_TO_OTEL_PROVIDER = {
    "openai":    GenAISystem.OPENAI.value,
    "anthropic": GenAISystem.ANTHROPIC.value,
    "gemini":    GenAiSystemValues.GCP_GEMINI.value,
    "vertex_ai": GenAiSystemValues.GCP_VERTEX_AI.value,
    "bedrock":   GenAISystem.AWS.value,
    "azure":     GenAiSystemValues.AZURE_AI_OPENAI.value,
    "groq":      GenAISystem.GROQ.value,
    "mistral":   GenAISystem.MISTRALAI.value,
    "cohere":    GenAISystem.COHERE.value,
    "ollama":    GenAISystem.OLLAMA.value,
}

_MODEL_PATTERN_TO_OTEL_PROVIDER = [
    ("claude",  GenAISystem.ANTHROPIC.value),
    ("gemini",  GenAiSystemValues.GCP_GEMINI.value),
    ("mistral", GenAISystem.MISTRALAI.value),
    ("command", GenAISystem.COHERE.value),
]


def _infer_provider(model: object | None) -> str | None:
    if not model:
        return None
    s = str(model).strip()
    if "/" in s:
        return _LITELLM_PREFIX_TO_OTEL_PROVIDER.get(s.split("/")[0].lower())
    lower = s.lower()
    if lower.startswith(("gpt-", "o1", "o3", "o4")):
        return GenAISystem.OPENAI.value
    for pattern, provider in _MODEL_PATTERN_TO_OTEL_PROVIDER:
        if pattern in lower:
            return provider
    return None


class DSPyInstrumentor(BaseInstrumentor):

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        token_histogram = duration_histogram = None
        if is_metrics_enabled():
            token_histogram, duration_histogram = _create_metrics(meter)

        wrap_function_wrapper(
            "dspy.clients.lm", "LM.forward",
            wrap_lm_forward(tracer, duration_histogram, token_histogram),
        )
        wrap_function_wrapper(
            "dspy.clients.lm", "LM.aforward",
            wrap_lm_aforward(tracer, duration_histogram, token_histogram),
        )
        wrap_function_wrapper(
            "dspy.predict.predict", "Predict.forward",
            wrap_predict_forward(tracer),
        )
        wrap_function_wrapper(
            "dspy.predict.predict", "Predict.aforward",
            wrap_predict_aforward(tracer),
        )

    def _uninstrument(self, **kwargs):
        unwrap("dspy.clients.lm.LM", "forward")
        unwrap("dspy.clients.lm.LM", "aforward")
        unwrap("dspy.predict.predict.Predict", "forward")
        unwrap("dspy.predict.predict.Predict", "aforward")


def with_tracer_wrapper(func):
    def _with_tracer(tracer, duration_histogram, token_histogram):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs)
        return wrapper
    return _with_tracer


def with_tracer_async_wrapper(func):
    def _with_tracer(tracer, duration_histogram, token_histogram):
        async def wrapper(wrapped, instance, args, kwargs):
            return await func(tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs)
        return wrapper
    return _with_tracer


def with_predict_tracer_wrapper(func):
    def _with_tracer(tracer):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, wrapped, instance, args, kwargs)
        return wrapper
    return _with_tracer


def with_predict_tracer_async_wrapper(func):
    def _with_tracer(tracer):
        async def wrapper(wrapped, instance, args, kwargs):
            return await func(tracer, wrapped, instance, args, kwargs)
        return wrapper
    return _with_tracer


def _lm_span_attrs(model: object | None, provider: str | None) -> dict:
    attrs = {GenAIAttributes.GEN_AI_OPERATION_NAME: GenAiOperationNameValues.CHAT.value}
    if model:
        attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] = str(model)
    if provider:
        attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] = provider
    return attrs


def _lm_span_name(instance, model: object | None) -> str:
    op = GenAiOperationNameValues.CHAT.value
    if model:
        return f"{op} {model}"
    return f"{op} {type(instance).__name__}"


def _extract_messages(args, kwargs):
    messages = kwargs.get("messages")
    if messages:
        return messages
    prompt = kwargs.get("prompt")
    if prompt:
        return [{"role": "user", "content": prompt}]
    return None


@with_tracer_wrapper
def wrap_lm_forward(tracer: Tracer, duration_histogram: Histogram, token_histogram: Histogram,
                    wrapped, instance, args, kwargs):
    model = getattr(instance, "model", None)
    provider = _infer_provider(model)

    with tracer.start_as_current_span(
        _lm_span_name(instance, model), kind=SpanKind.CLIENT, attributes=_lm_span_attrs(model, provider),
    ) as span:
        messages = _extract_messages(args, kwargs)
        input_json = messages_to_otel_input(messages)
        if input_json:
            set_span_attribute(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES, input_json)

        start = time.time()
        try:
            result = wrapped(*args, **kwargs)
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise
        _safe_set_lm_span_output(span, result, model, provider, duration_histogram, token_histogram, start)
        span.set_status(Status(StatusCode.OK))
        return result


@with_tracer_async_wrapper
async def wrap_lm_aforward(tracer: Tracer, duration_histogram: Histogram, token_histogram: Histogram,
                           wrapped, instance, args, kwargs):
    model = getattr(instance, "model", None)
    provider = _infer_provider(model)

    with tracer.start_as_current_span(
        _lm_span_name(instance, model), kind=SpanKind.CLIENT, attributes=_lm_span_attrs(model, provider),
    ) as span:
        messages = _extract_messages(args, kwargs)
        input_json = messages_to_otel_input(messages)
        if input_json:
            set_span_attribute(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES, input_json)

        start = time.time()
        try:
            result = await wrapped(*args, **kwargs)
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise
        _safe_set_lm_span_output(span, result, model, provider, duration_histogram, token_histogram, start)
        span.set_status(Status(StatusCode.OK))
        return result


def _safe_set_lm_span_output(span, result, model, provider,
                             duration_histogram, token_histogram, start):
    """Telemetry must never break the user's call; swallow post-call errors."""
    try:
        _set_lm_span_output(span, result, model, provider, duration_histogram, token_histogram, start)
    except Exception:
        logger.debug("Failed to set LM span output", exc_info=True)


@with_predict_tracer_wrapper
def wrap_predict_forward(tracer: Tracer, wrapped, instance, args, kwargs):
    sig_name = _signature_name(instance)
    with tracer.start_as_current_span(
        f"{sig_name}.predict",
        kind=SpanKind.INTERNAL,
        attributes={SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.TASK.value},
    ) as span:
        span.set_attribute(_DSPY_SIGNATURE, sig_name)
        try:
            result = wrapped(*args, **kwargs)
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise
        span.set_status(Status(StatusCode.OK))
        return result


@with_predict_tracer_async_wrapper
async def wrap_predict_aforward(tracer: Tracer, wrapped, instance, args, kwargs):
    sig_name = _signature_name(instance)
    with tracer.start_as_current_span(
        f"{sig_name}.predict",
        kind=SpanKind.INTERNAL,
        attributes={SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.TASK.value},
    ) as span:
        span.set_attribute(_DSPY_SIGNATURE, sig_name)
        try:
            result = await wrapped(*args, **kwargs)
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise
        span.set_status(Status(StatusCode.OK))
        return result


def _set_lm_span_output(span, result, model, provider,
                        duration_histogram, token_histogram, start):
    output_json = response_to_otel_output(result)
    if output_json:
        set_span_attribute(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, output_json)

    response_model = getattr(result, "model", None) or model
    set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, response_model)

    prompt_tokens, completion_tokens = get_token_usage(result)
    if prompt_tokens is not None:
        set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)
    if completion_tokens is not None:
        set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens)

    cache_hit = getattr(result, "cache_hit", False)
    if cache_hit:
        span.set_attribute(_DSPY_CACHE_HIT, True)

    metric_attrs = {}
    if response_model:
        metric_attrs[GenAIAttributes.GEN_AI_RESPONSE_MODEL] = str(response_model)
    if provider:
        metric_attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] = provider

    if duration_histogram:
        duration_histogram.record(time.time() - start, attributes=metric_attrs)
    if token_histogram:
        if prompt_tokens is not None:
            token_histogram.record(
                prompt_tokens,
                attributes={**metric_attrs, GenAIAttributes.GEN_AI_TOKEN_TYPE: "input"},
            )
        if completion_tokens is not None:
            token_histogram.record(
                completion_tokens,
                attributes={**metric_attrs, GenAIAttributes.GEN_AI_TOKEN_TYPE: "output"},
            )


def _signature_name(predict_instance) -> str:
    cls_name = type(predict_instance).__name__
    sig = getattr(predict_instance, "signature", None)
    if sig is not None:
        sig_name = getattr(sig, "__name__", None)
        if sig_name and sig_name != "Signature":
            return sig_name
        fields = getattr(sig, "fields", None)
        if fields:
            return "_".join(fields.keys())
    return cls_name


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
