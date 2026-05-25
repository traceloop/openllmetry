import os
import time
from contextvars import ContextVar
from typing import Collection

from wrapt import wrap_function_wrapper
from opentelemetry.trace import SpanKind, get_current_span, get_tracer, Tracer
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
    GenAiSystemValues,
)
from opentelemetry.semconv_ai import GenAISystem, SpanAttributes, TraceloopSpanKindValues, Meters
from .crewai_span_attributes import CrewAISpanAttributes, set_span_attribute
from .utils import _messages_to_otel_input, _response_to_otel_output

_instruments = ("crewai >= 1.0.0",)
_GEN_AI_OPERATION_PLAN = "plan"
_crew_planning_span_active = ContextVar("crew_planning_span_active", default=False)

# Maps LiteLLM vendor prefixes (e.g. "openai" in "openai/gpt-4") to OTel provider name values.
# Uses GenAISystem (semconv-ai) and GenAiSystemValues (OTel upstream) — no raw strings.
_LITELLM_PREFIX_TO_OTEL_PROVIDER = {
    "openai":      GenAISystem.OPENAI.value,
    "anthropic":   GenAISystem.ANTHROPIC.value,
    "gemini":      GenAiSystemValues.GCP_GEMINI.value,
    "vertex_ai":   GenAiSystemValues.GCP_VERTEX_AI.value,
    "bedrock":     GenAISystem.AWS.value,
    "azure":       GenAiSystemValues.AZURE_AI_OPENAI.value,
    "groq":        GenAISystem.GROQ.value,
    "mistral":     GenAISystem.MISTRALAI.value,
    "cohere":      GenAISystem.COHERE.value,
    "ollama":      GenAISystem.OLLAMA.value,
}

# Maps bare model name patterns to OTel provider name values.
_MODEL_PATTERN_TO_OTEL_PROVIDER = [
    ("claude",   GenAISystem.ANTHROPIC.value),
    ("gemini",   GenAiSystemValues.GCP_GEMINI.value),
    ("mistral",  GenAISystem.MISTRALAI.value),
    ("command",  GenAISystem.COHERE.value),
]


def _infer_llm_provider_from_model(model: object | None) -> str | None:
    """Resolve gen_ai.provider.name for the underlying LLM on a chat span.

    LiteLLM-prefixed strings ("openai/gpt-4") use the prefix via
    _LITELLM_PREFIX_TO_OTEL_PROVIDER. Bare model names use pattern matching
    via _MODEL_PATTERN_TO_OTEL_PROVIDER. Returns None when unknown — never guesses.
    """
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
        _wrap_optional_function_wrapper(
            "crewai.utilities.planning_handler",
            "CrewPlanner._handle_crew_planning",
            wrap_crew_planning(tracer, duration_histogram, token_histogram),
        )
        _wrap_optional_function_wrapper(
            "crewai.utilities.planning_handler",
            "CrewPlanner._create_planning_agent",
            wrap_create_planning_agent,
        )

    def _uninstrument(self, **kwargs):
        unwrap("crewai.crew.Crew", "kickoff")
        unwrap("crewai.agent.Agent", "execute_task")
        unwrap("crewai.task.Task", "execute_sync")
        unwrap("crewai.llm.LLM", "call")
        _unwrap_optional("crewai.utilities.planning_handler.CrewPlanner", "_handle_crew_planning")
        _unwrap_optional("crewai.utilities.planning_handler.CrewPlanner", "_create_planning_agent")


def _wrap_optional_function_wrapper(module, name, wrapper):
    try:
        wrap_function_wrapper(module, name, wrapper)
    except (AttributeError, ImportError, ModuleNotFoundError):
        pass


def _unwrap_optional(module, name):
    try:
        unwrap(module, name)
    except (AttributeError, ImportError, ModuleNotFoundError):
        pass


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
            GenAIAttributes.GEN_AI_PROVIDER_NAME: GenAISystem.CREWAI.value,
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
        kind=SpanKind.INTERNAL,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.AGENT.value,
            GenAIAttributes.GEN_AI_PROVIDER_NAME: GenAISystem.CREWAI.value,
            GenAIAttributes.GEN_AI_OPERATION_NAME: GenAiOperationNameValues.INVOKE_AGENT.value,
        }
    ) as span:
        try:
            CrewAISpanAttributes(span=span, instance=instance)
            if hasattr(instance, "role") and instance.role:
                set_span_attribute(span, GenAIAttributes.GEN_AI_AGENT_NAME, instance.role)
            if hasattr(instance, "id"):
                set_span_attribute(span, GenAIAttributes.GEN_AI_AGENT_ID, str(instance.id))
            result = wrapped(*args, **kwargs)
            if token_histogram:
                token_histogram.record(
                    instance._token_process.get_summary().prompt_tokens,
                    attributes={
                        GenAIAttributes.GEN_AI_PROVIDER_NAME: GenAISystem.CREWAI.value,
                        GenAIAttributes.GEN_AI_TOKEN_TYPE: "input",
                        GenAIAttributes.GEN_AI_RESPONSE_MODEL: str(instance.llm.model),
                    }
                )
                token_histogram.record(
                    instance._token_process.get_summary().completion_tokens,
                    attributes={
                        GenAIAttributes.GEN_AI_PROVIDER_NAME: GenAISystem.CREWAI.value,
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
        kind=SpanKind.INTERNAL,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.TASK.value,
            GenAIAttributes.GEN_AI_PROVIDER_NAME: GenAISystem.CREWAI.value,
            GenAIAttributes.GEN_AI_OPERATION_NAME: GenAiOperationNameValues.INVOKE_AGENT.value,
        }
    ) as span:
        try:
            CrewAISpanAttributes(span=span, instance=instance)
            result = wrapped(*args, **kwargs)
            set_span_attribute(span, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, str(result))
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


@with_tracer_wrapper
def wrap_crew_planning(tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs):
    with tracer.start_as_current_span(
        _GEN_AI_OPERATION_PLAN,
        kind=SpanKind.INTERNAL,
        attributes={
            GenAIAttributes.GEN_AI_PROVIDER_NAME: GenAISystem.CREWAI.value,
            GenAIAttributes.GEN_AI_OPERATION_NAME: _GEN_AI_OPERATION_PLAN,
        },
    ) as span:
        token = _crew_planning_span_active.set(True)
        try:
            result = wrapped(*args, **kwargs)
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise
        finally:
            _crew_planning_span_active.reset(token)


def wrap_create_planning_agent(wrapped, instance, args, kwargs):
    planner_agent = wrapped(*args, **kwargs)
    if _crew_planning_span_active.get():
        span = get_current_span()
        agent_name = getattr(planner_agent, "role", None)
        agent_id = getattr(planner_agent, "id", None)
        if agent_name:
            set_span_attribute(span, GenAIAttributes.GEN_AI_AGENT_NAME, agent_name)
            span.update_name(f"plan {agent_name}")
        if agent_id:
            set_span_attribute(span, GenAIAttributes.GEN_AI_AGENT_ID, str(agent_id))
    return planner_agent


@with_tracer_wrapper
def wrap_llm_call(tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs):
    model = str(instance.model) if hasattr(instance, "model") else "llm"
    provider = _infer_llm_provider_from_model(getattr(instance, "model", None))

    span_attrs = {
        GenAIAttributes.GEN_AI_OPERATION_NAME: GenAiOperationNameValues.CHAT.value,
        GenAIAttributes.GEN_AI_REQUEST_MODEL: model,
    }
    if provider:
        span_attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] = provider

    with tracer.start_as_current_span(
        f"{model}.llm", kind=SpanKind.CLIENT, attributes=span_attrs,
    ) as span:
        start_time = time.time()
        try:
            CrewAISpanAttributes(span=span, instance=instance)
            messages_arg = args[0] if args else kwargs.get("messages")
            result = wrapped(*args, **kwargs)

            _set_messages_attributes(span, messages_arg, result)
            _set_response_attributes(span, instance)
            _record_duration(duration_histogram, start_time, model, provider)

            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


def _set_messages_attributes(span, messages_arg, result):
    input_json = _messages_to_otel_input(messages_arg)
    if input_json:
        set_span_attribute(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES, input_json)
    output_json = _response_to_otel_output(result)
    if output_json:
        set_span_attribute(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, output_json)


def _set_response_attributes(span, instance):
    set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, str(instance.model))
    if hasattr(instance, "last_token_usage") and instance.last_token_usage:
        usage = instance.last_token_usage
        set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
                           getattr(usage, "prompt_tokens", None))
        set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                           getattr(usage, "completion_tokens", None))


def _record_duration(duration_histogram, start_time, model, provider):
    if not duration_histogram:
        return
    attrs = {GenAIAttributes.GEN_AI_RESPONSE_MODEL: model}
    if provider:
        attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] = provider
    duration_histogram.record(time.time() - start_time, attributes=attrs)


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
