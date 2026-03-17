import json
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
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues, Meters
from .crewai_span_attributes import CrewAISpanAttributes, set_span_attribute

_instruments = ("crewai >= 0.70.0",)


def _safe_json(obj):
    """Serialize obj to JSON string, falling back to str()."""
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return str(obj)


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
        wrap_function_wrapper("crewai.tools.tool_usage", "ToolUsage._use",
                              wrap_tool_use(tracer))

    def _uninstrument(self, **kwargs):
        unwrap("crewai.crew.Crew", "kickoff")
        unwrap("crewai.agent.Agent", "execute_task")
        unwrap("crewai.task.Task", "execute_sync")
        unwrap("crewai.llm.LLM", "call")
        unwrap("crewai.tools.tool_usage.ToolUsage", "_use")


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
    crew_name = getattr(instance, "name", None) or "Crew"
    with tracer.start_as_current_span(
        crew_name,
        kind=SpanKind.INTERNAL,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.WORKFLOW.value,
        }
    ) as span:
        try:
            CrewAISpanAttributes(span=span, instance=instance)

            # Set inputs: task descriptions and agent roles
            inputs_data = kwargs.get("inputs", {})
            if args:
                inputs_data = args[0] if isinstance(args[0], dict) else inputs_data
            tasks_summary = []
            for task in getattr(instance, "tasks", []):
                tasks_summary.append({
                    "description": getattr(task, "description", ""),
                    "agent": getattr(task.agent, "role", "") if task.agent else "",
                })
            crew_input = {
                "inputs": inputs_data,
                "tasks": tasks_summary,
                "agents": [getattr(a, "role", "") for a in getattr(instance, "agents", [])],
            }
            set_span_attribute(span, SpanAttributes.TRACELOOP_ENTITY_INPUT, _safe_json(crew_input))

            result = wrapped(*args, **kwargs)

            if result:
                set_span_attribute(span, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, str(result))
                span.set_status(Status(StatusCode.OK))
                if instance.__class__.__name__ == "Crew":
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
        f"{agent_name}",
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.AGENT.value,
        }
    ) as span:
        try:
            CrewAISpanAttributes(span=span, instance=instance)

            # Set input: the task being executed and context
            task = args[0] if args else kwargs.get("task")
            context = args[1] if len(args) > 1 else kwargs.get("context")
            if task:
                agent_input = {
                    "task": getattr(task, "description", str(task)),
                    "expected_output": getattr(task, "expected_output", ""),
                }
                if context:
                    agent_input["context"] = str(context)[:500]
                tools_list = args[2] if len(args) > 2 else kwargs.get("tools")
                if tools_list:
                    agent_input["tools"] = [getattr(t, "name", str(t)) for t in tools_list]
                set_span_attribute(span, SpanAttributes.TRACELOOP_ENTITY_INPUT,
                                   _safe_json(agent_input))

            result = wrapped(*args, **kwargs)

            if result:
                set_span_attribute(span, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, str(result))

            if token_histogram:
                token_histogram.record(
                    instance._token_process.get_summary().prompt_tokens,
                    attributes={
                        GenAIAttributes.GEN_AI_SYSTEM: "crewai",
                        GenAIAttributes.GEN_AI_TOKEN_TYPE: "input",
                        GenAIAttributes.GEN_AI_RESPONSE_MODEL: str(instance.llm.model),
                    }
                )
                token_histogram.record(
                    instance._token_process.get_summary().completion_tokens,
                    attributes={
                        GenAIAttributes.GEN_AI_SYSTEM: "crewai",
                        GenAIAttributes.GEN_AI_TOKEN_TYPE: "output",
                        GenAIAttributes.GEN_AI_RESPONSE_MODEL: str(instance.llm.model),
                    },
                )

            # Store model as metadata but NOT as gen_ai.request.model
            # (that would cause the backend to classify this as run_type=llm)
            span.set_attribute("crewai.agent.model", str(instance.llm.model))
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


@with_tracer_wrapper
def wrap_task_execute(tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs):
    task_name = getattr(instance, "name", None)
    if not task_name:
        desc = getattr(instance, "description", "task")
        task_name = desc[:60] + "..." if len(desc) > 60 else desc

    with tracer.start_as_current_span(
        task_name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.TASK.value,
        }
    ) as span:
        try:
            CrewAISpanAttributes(span=span, instance=instance)

            # Set input: task description and expected output
            task_input = {
                "description": getattr(instance, "description", ""),
                "expected_output": getattr(instance, "expected_output", ""),
                "agent": getattr(instance.agent, "role", "") if instance.agent else "",
            }
            set_span_attribute(span, SpanAttributes.TRACELOOP_ENTITY_INPUT, _safe_json(task_input))

            result = wrapped(*args, **kwargs)
            set_span_attribute(span, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, str(result))
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


def wrap_tool_use(tracer):
    """Wrap CrewAI ToolUsage._use to create tool spans with inputs/outputs."""
    def wrapper(wrapped, instance, args, kwargs):
        # instance is ToolUsage, args/kwargs contain tool_string, tool, calling
        calling = kwargs.get("calling") or (args[1] if len(args) > 1 else None)
        tool = kwargs.get("tool") or (args[0] if args else None)

        tool_name = "unknown_tool"
        tool_input = {}
        if calling:
            tool_name = getattr(calling, "tool_name", tool_name)
            tool_input = getattr(calling, "arguments", {}) or {}
        if tool and hasattr(tool, "name"):
            tool_name = tool.name

        with tracer.start_as_current_span(
            tool_name,
            kind=SpanKind.INTERNAL,
            attributes={
                SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.TOOL.value,
            }
        ) as span:
            set_span_attribute(span, SpanAttributes.TRACELOOP_ENTITY_INPUT, _safe_json(tool_input))
            if tool and hasattr(tool, "description"):
                span.set_attribute("tool.description", str(tool.description))

            try:
                result = wrapped(*args, **kwargs)
                set_span_attribute(span, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, str(result))
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as ex:
                span.set_status(Status(StatusCode.ERROR, str(ex)))
                raise

    return wrapper


@with_tracer_wrapper
def wrap_llm_call(tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs):
    llm = instance.model if hasattr(instance, "model") else "llm"
    with tracer.start_as_current_span(
        f"{llm}.llm",
        kind=SpanKind.CLIENT,
        attributes={
        }
    ) as span:
        start_time = time.time()
        try:
            CrewAISpanAttributes(span=span, instance=instance)
            result = wrapped(*args, **kwargs)

            if duration_histogram:
                duration = time.time() - start_time
                duration_histogram.record(
                    duration,
                    attributes={
                     GenAIAttributes.GEN_AI_SYSTEM: "crewai",
                     GenAIAttributes.GEN_AI_RESPONSE_MODEL: str(instance.model)
                    },
                )

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
