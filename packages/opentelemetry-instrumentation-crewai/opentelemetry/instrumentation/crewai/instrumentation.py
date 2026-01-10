import json
import os
import time
from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.metrics import Histogram, Meter, get_meter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import Meters, SpanAttributes, TraceloopSpanKindValues
from opentelemetry.trace import SpanKind, Tracer, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.crewai.version import __version__

from .crewai_span_attributes import CrewAISpanAttributes, set_span_attribute

_instruments = ("crewai >= 0.70.0",)


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

        wrap_function_wrapper(
            "crewai.crew",
            "Crew.kickoff",
            wrap_kickoff(tracer, duration_histogram, token_histogram),
        )
        wrap_function_wrapper(
            "crewai.agent",
            "Agent.execute_task",
            wrap_agent_execute_task(tracer, duration_histogram, token_histogram),
        )
        wrap_function_wrapper(
            "crewai.task",
            "Task.execute_sync",
            wrap_task_execute(tracer, duration_histogram, token_histogram),
        )
        wrap_function_wrapper(
            "crewai.llm",
            "LLM.call",
            wrap_llm_call(tracer, duration_histogram, token_histogram),
        )
        wrap_function_wrapper(
            "crewai.tools.tool_usage",
            "ToolUsage._use",
            wrap_tool_use(tracer, duration_histogram, token_histogram),
        )

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
            return func(
                tracer,
                duration_histogram,
                token_histogram,
                wrapped,
                instance,
                args,
                kwargs,
            )

        return wrapper

    return _with_tracer


@with_tracer_wrapper
def wrap_kickoff(
    tracer: Tracer,
    duration_histogram: Histogram,
    token_histogram: Histogram,
    wrapped,
    instance,
    args,
    kwargs,
):
    with tracer.start_as_current_span(
        "crewai.workflow",
        kind=SpanKind.INTERNAL,
        attributes={
            GenAIAttributes.GEN_AI_SYSTEM: "crewai",
        },
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
                            span.set_attribute(
                                f"crewai.crew.{attr}", str(getattr(result, attr))
                            )
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


@with_tracer_wrapper
def wrap_agent_execute_task(
    tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs
):
    agent_name = instance.role if hasattr(instance, "role") else "agent"
    with tracer.start_as_current_span(
        f"{agent_name}.agent",
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.AGENT.value,
            GenAIAttributes.GEN_AI_OPERATION_NAME: GenAIAttributes.GenAiOperationNameValues.INVOKE_AGENT.value,
        },
    ) as span:
        try:
            CrewAISpanAttributes(span=span, instance=instance)
            result = wrapped(*args, **kwargs)
            if token_histogram:
                token_histogram.record(
                    instance._token_process.get_summary().prompt_tokens,
                    attributes={
                        GenAIAttributes.GEN_AI_SYSTEM: "crewai",
                        GenAIAttributes.GEN_AI_TOKEN_TYPE: "input",
                        GenAIAttributes.GEN_AI_RESPONSE_MODEL: str(instance.llm.model),
                    },
                )
                token_histogram.record(
                    instance._token_process.get_summary().completion_tokens,
                    attributes={
                        GenAIAttributes.GEN_AI_SYSTEM: "crewai",
                        GenAIAttributes.GEN_AI_TOKEN_TYPE: "output",
                        GenAIAttributes.GEN_AI_RESPONSE_MODEL: str(instance.llm.model),
                    },
                )

            set_span_attribute(
                span, GenAIAttributes.GEN_AI_REQUEST_MODEL, str(instance.llm.model)
            )
            set_span_attribute(
                span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, str(instance.llm.model)
            )
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


@with_tracer_wrapper
def wrap_task_execute(
    tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs
):
    task_name = instance.description if hasattr(instance, "description") else "task"

    with tracer.start_as_current_span(
        f"{task_name}.task",
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.TASK.value,
        },
    ) as span:
        try:
            CrewAISpanAttributes(span=span, instance=instance)
            result = wrapped(*args, **kwargs)
            set_span_attribute(
                span, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, str(result)
            )
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


@with_tracer_wrapper
def wrap_llm_call(
    tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs
):
    llm = instance.model if hasattr(instance, "model") else "llm"
    with tracer.start_as_current_span(
        f"{llm}.llm",
        kind=SpanKind.CLIENT,
        attributes={
            GenAIAttributes.GEN_AI_OPERATION_NAME: GenAIAttributes.GenAiOperationNameValues.CHAT.value,
            GenAIAttributes.GEN_AI_INPUT_MESSAGES: json.dumps(args),
        },
    ) as span:
        start_time = time.time()
        try:
            CrewAISpanAttributes(span=span, instance=instance)
            result = wrapped(*args, **kwargs)
            span.set_attribute(
                GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
                json.dumps([{"role": "assistant", "content": result}]),
            )

            if duration_histogram:
                duration = time.time() - start_time
                duration_histogram.record(
                    duration,
                    attributes={
                        GenAIAttributes.GEN_AI_SYSTEM: "crewai",
                        GenAIAttributes.GEN_AI_RESPONSE_MODEL: str(instance.model),
                    },
                )

            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


@with_tracer_wrapper
def wrap_tool_use(
    tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs
):
    tool = kwargs.get("tool")
    tool_name = "unknown"
    attributes: dict[str, Any] = {
        GenAIAttributes.GEN_AI_OPERATION_NAME: GenAIAttributes.GenAiOperationNameValues.EXECUTE_TOOL.value,
    }
    if tool:
        tool_name = tool.name
        attributes.update(
            {
                GenAIAttributes.GEN_AI_TOOL_NAME: tool_name,
                "crewai.tool.current_usage_count": getattr(
                    tool, "current_usage_count", 0
                ),
                "crewai.tool.max_usage_count": getattr(tool, "max_usage_count", -1),
                "crewai.tool.result_as_answer": getattr(
                    tool, "result_as_answer", False
                ),
                GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS: json.dumps(
                    getattr(tool, "args", {})
                ),
            }
        )
    with tracer.start_as_current_span(
        f"{tool_name}.tool", kind=SpanKind.CLIENT, attributes=attributes
    ) as span:
        start_time = time.time()
        try:
            response = wrapped(*args, **kwargs)
            if duration_histogram:
                duration = time.time() - start_time
                duration_histogram.record(
                    duration,
                    attributes={
                        GenAIAttributes.GEN_AI_SYSTEM: "crewai",
                        GenAIAttributes.GEN_AI_TOOL_NAME: tool_name,
                    },
                )

        except Exception as exception:
            span.set_status(Status(StatusCode.ERROR, str(exception)))
            span.record_exception(exception)
            raise
        span.set_status(Status(StatusCode.OK))
        span.set_attribute(
            GenAIAttributes.GEN_AI_TOOL_CALL_RESULT, json.dumps(response)
        )
        return response


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
