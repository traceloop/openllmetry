from typing import Collection
from wrapt import wrap_function_wrapper
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import SpanKind, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.instrumentation.crewai.version import __version__
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues
from .crewai_span_attributes import CrewAISpanAttributes,set_span_attribute

_instruments = ("crewai >= 0.70.0",)

class CrewAIInstrumentor(BaseInstrumentor):


    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments


    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        wrap_function_wrapper("crewai.crew", "Crew.kickoff", wrap_kickoff(tracer))
        wrap_function_wrapper("crewai.agent", "Agent.execute_task", wrap_agent_execute_task(tracer))
        wrap_function_wrapper("crewai.task", "Task.execute_sync", wrap_task_execute(tracer))
        wrap_function_wrapper("crewai.llm", "LLM.call", wrap_llm_call(tracer))
    
    def _uninstrument(self, **kwargs):
        pass


def with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, wrapped, instance, args, kwargs)
        return wrapper
    return _with_tracer


@with_tracer_wrapper
def wrap_kickoff(tracer, wrapped, instance, args, kwargs):
    with tracer.start_as_current_span(
        "crewai.workflow",
        kind=SpanKind.INTERNAL,
        attributes={
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
def wrap_agent_execute_task(tracer, wrapped, instance, args, kwargs):
    agent_name = instance.role if hasattr(instance, "role") else "agent"
    with tracer.start_as_current_span(
        f"{agent_name}.agent",
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.AGENT.value,
        }
    ) as span:
        try:
            CrewAISpanAttributes(span=span, instance=instance)
            result = wrapped(*args, **kwargs)
            class_name = instance.__class__.__name__
            span.set_attribute(f"crewai.{class_name.lower()}.model", str(instance.llm.model))
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise
        
@with_tracer_wrapper
def wrap_task_execute(tracer, wrapped, instance, args, kwargs):
    task_name = instance.description if hasattr(instance, "description") else "task"

    with tracer.start_as_current_span(
        f"Task.execute.{task_name}",
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.TASK.value,
        }
    ) as span:
        try:
            CrewAISpanAttributes(span=span, instance=instance)
            result = wrapped(*args, **kwargs)
            class_name = instance.__class__.__name__
            span.set_attribute(f"crewai.{class_name.lower()}.result", str(result))
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise 


@with_tracer_wrapper
def wrap_llm_call(tracer, wrapped, instance, args, kwargs):
    llm = instance.model if hasattr(instance, "model") else "llm"
    with tracer.start_as_current_span(
        f"LLM.{llm}",
        kind=SpanKind.CLIENT,
        attributes={
        }
    ) as span:
        try:
            CrewAISpanAttributes(span=span, instance=instance)
            result = wrapped(*args, **kwargs)
            set_span_attribute(span,SpanAttributes.LLM_SYSTEM, "litellm")
            set_span_attribute(span,SpanAttributes.LLM_REQUEST_MODEL, str(instance.model))
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise
        