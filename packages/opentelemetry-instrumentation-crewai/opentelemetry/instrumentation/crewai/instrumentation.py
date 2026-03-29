import os
import time
from typing import Collection

from wrapt import wrap_function_wrapper
from opentelemetry.trace import SpanKind, get_tracer, Tracer
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.metrics import Histogram, Meter, get_meter
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.crewai.version import __version__
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiOperationNameValues,
)
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues, Meters
from .crewai_span_attributes import CrewAISpanAttributes, set_span_attribute
from .utils import should_send_prompts, _messages_to_otel_input, _response_to_otel_output

_instruments = ("crewai >= 1.0.0",)


def _infer_llm_provider_from_model(model: object | None) -> str | None:
    """Best-effort gen_ai.provider.name for the underlying LLM (chat span), not CrewAI itself."""
    if model is None:
        return None
    s = str(model).strip()
    if not s:
        return None
    lower = s.lower()
    if "/" in lower:
        vendor, _ = lower.split("/", 1)
        vendor_aliases = {
            "openai": "openai",
            "anthropic": "anthropic",
            "azure": "azure.ai.openai",
            "google": "gcp.gen_ai",
            "gemini": "gcp.gen_ai",
            "vertex_ai": "gcp.vertex_ai",
            "bedrock": "aws.bedrock",
            "groq": "groq",
            "mistral": "mistral_ai",
            "cohere": "cohere",
            "ollama": "ollama",
        }
        if vendor in vendor_aliases:
            return vendor_aliases[vendor]
    if (
        lower.startswith("gpt-")
        or "gpt-3" in lower
        or "gpt-4" in lower
        or lower.startswith("o1")
        or lower.startswith("o3")
    ):
        return "openai"
    if "claude" in lower:
        return "anthropic"
    if "gemini" in lower:
        return "gcp.gen_ai"
    return None


class CrewAIInstrumentor(BaseInstrumentor):

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        token_histogram = None
        duration_histogram = None
        if is_metrics_enabled():
            token_histogram, duration_histogram = _create_metrics(meter)

        wrap_function_wrapper("crewai.crew", "Crew.kickoff",
                              wrap_kickoff(tracer, duration_histogram, token_histogram))
        wrap_function_wrapper("crewai.agent", "Agent.execute_task",
                              wrap_agent_execute_task(tracer, duration_histogram, token_histogram))
        wrap_function_wrapper("crewai.task", "Task.execute_sync",
                              wrap_task_execute(tracer, duration_histogram, token_histogram))
        wrap_function_wrapper("crewai.llm", "LLM.call",
                              wrap_llm_call(tracer, duration_histogram, token_histogram))

    def _uninstrument(self, **kwargs):
        unwrap("crewai.crew.Crew", "kickoff")
        unwrap("crewai.agent.Agent", "execute_task")
        unwrap("crewai.task.Task", "execute_sync")
        unwrap("crewai.llm.LLM", "call")


def with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, duration_histogram, token_histogram):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs)
        return wrapper
    return _with_tracer


@with_tracer_wrapper
def wrap_kickoff(tracer: Tracer, duration_histogram: Histogram, token_histogram: Histogram,
                 wrapped, instance, args, kwargs):
    with tracer.start_as_current_span(
        "crewai.workflow",
        kind=SpanKind.INTERNAL,
        attributes={
            GenAIAttributes.GEN_AI_PROVIDER_NAME: "crewai",
            GenAIAttributes.GEN_AI_OPERATION_NAME: GenAiOperationNameValues.INVOKE_AGENT.value,
        }
    ) as span:
        try:
            CrewAISpanAttributes(span=span, instance=instance)
            result = wrapped(*args, **kwargs)
            if result:
                class_name = instance.__class__.__name__
                span.set_attribute(f"crewai.{class_name.lower()}.result", str(result))
                span.set_status(Status(StatusCode.OK))
                if class_name == "Crew":
                    for attr in ["tasks_output", "token_usage", "usage_metrics"]:
                        if hasattr(result, attr):
                            span.set_attribute(f"crewai.crew.{attr}", str(getattr(result, attr)))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


@with_tracer_wrapper
def wrap_agent_execute_task(tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs):
    agent_name = instance.role if hasattr(instance, "role") else "agent"
    with tracer.start_as_current_span(
        f"{agent_name}.agent",
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.AGENT.value,
            GenAIAttributes.GEN_AI_PROVIDER_NAME: "crewai",
            GenAIAttributes.GEN_AI_OPERATION_NAME: GenAiOperationNameValues.INVOKE_AGENT.value,
        }
    ) as span:
        try:
            CrewAISpanAttributes(span=span, instance=instance)
            result = wrapped(*args, **kwargs)
            if token_histogram:
                token_histogram.record(
                    instance._token_process.get_summary().prompt_tokens,
                    attributes={
                        GenAIAttributes.GEN_AI_PROVIDER_NAME: "crewai",
                        GenAIAttributes.GEN_AI_TOKEN_TYPE: "input",
                        GenAIAttributes.GEN_AI_RESPONSE_MODEL: str(instance.llm.model),
                    }
                )
                token_histogram.record(
                    instance._token_process.get_summary().completion_tokens,
                    attributes={
                        GenAIAttributes.GEN_AI_PROVIDER_NAME: "crewai",
                        GenAIAttributes.GEN_AI_TOKEN_TYPE: "output",
                        GenAIAttributes.GEN_AI_RESPONSE_MODEL: str(instance.llm.model),
                    },
                )

            set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MODEL, str(instance.llm.model))
            set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, str(instance.llm.model))
            summary = instance._token_process.get_summary()
            if summary.prompt_tokens:
                set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, summary.prompt_tokens)
            if summary.completion_tokens:
                set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, summary.completion_tokens)
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


@with_tracer_wrapper
def wrap_task_execute(tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs):
    task_name = instance.description if hasattr(instance, "description") else "task"

    with tracer.start_as_current_span(
        f"{task_name}.task",
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.TASK.value,
            GenAIAttributes.GEN_AI_PROVIDER_NAME: "crewai",
            GenAIAttributes.GEN_AI_OPERATION_NAME: GenAiOperationNameValues.INVOKE_AGENT.value,
        }
    ) as span:
        try:
            CrewAISpanAttributes(span=span, instance=instance)
            result = wrapped(*args, **kwargs)
            if should_send_prompts():
                set_span_attribute(span, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, str(result))
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


@with_tracer_wrapper
def wrap_llm_call(tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs):
    llm = instance.model if hasattr(instance, "model") else "llm"
    chat_provider = _infer_llm_provider_from_model(getattr(instance, "model", None))
    span_attrs = {
        GenAIAttributes.GEN_AI_OPERATION_NAME: GenAiOperationNameValues.CHAT.value,
    }
    if chat_provider is not None:
        span_attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] = chat_provider

    with tracer.start_as_current_span(
        f"{llm}.llm",
        kind=SpanKind.CLIENT,
        attributes=span_attrs,
    ) as span:
        start_time = time.time()
        try:
            CrewAISpanAttributes(span=span, instance=instance)
            # Capture input messages before the call (args[0] is the messages param of LLM.call)
            messages_arg = args[0] if args else kwargs.get("messages")
            result = wrapped(*args, **kwargs)
            # Known gap: CrewAI LLM.call returns plain text; stop/finish reason is not exposed
            # for mapping to gen_ai.response.finish_reasons without provider-specific internals.

            if should_send_prompts():
                input_json = _messages_to_otel_input(messages_arg)
                if input_json:
                    set_span_attribute(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES, input_json)
                output_json = _response_to_otel_output(result)
                if output_json:
                    set_span_attribute(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, output_json)

            if duration_histogram:
                duration = time.time() - start_time
                metric_attrs = {
                    GenAIAttributes.GEN_AI_RESPONSE_MODEL: str(instance.model),
                }
                if chat_provider is not None:
                    metric_attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] = chat_provider
                duration_histogram.record(duration, attributes=metric_attrs)

            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


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
