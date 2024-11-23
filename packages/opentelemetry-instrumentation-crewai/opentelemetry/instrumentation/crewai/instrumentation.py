from typing import Collection
from wrapt import wrap_function_wrapper
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import SpanKind, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.instrumentation.crewai.version import __version__
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues

_instruments = ("crewai >= 0.70.0",)

class CrewAIInstrumentor(BaseInstrumentor):
    
    
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments
    
    
    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        wrap_function_wrapper("crewai", "Crew.kickoff", wrap_kickoff(tracer))
        wrap_function_wrapper("crewai.agent", "Agent.execute_task", wrap_agent_execute_task(tracer))
        wrap_function_wrapper("crewai.tool", "Tool.run", wrap_tool_run(tracer))
        # wrap_function_wrapper("crewai.task", "Task.execute", wrap_execute_task(tracer))

    
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
            result = wrapped(*args, **kwargs)
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


@with_tracer_wrapper
def wrap_agent_execute_task(tracer, wrapped, instance, args, kwargs):
    agent_name = instance.role if hasattr(instance, "role") else "agent"

    with tracer.start_as_current_span(
        f"{agent_name}.agent",
        kind=SpanKind.INTERNAL,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.AGENT.value,
        }
    ) as span:
        try:
            result = wrapped(*args, **kwargs)
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise 


@with_tracer_wrapper
def wrap_tool_run(tracer, wrapped, instance, args, kwargs):
    tool_name = instance.name if hasattr(instance, "name") else "tool"
    
    with tracer.start_as_current_span(
        f"{tool_name}.tool",
        kind=SpanKind.INTERNAL,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.TOOL.value,
        }
    ) as span:
        try:
            result = wrapped(*args, **kwargs)
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise 