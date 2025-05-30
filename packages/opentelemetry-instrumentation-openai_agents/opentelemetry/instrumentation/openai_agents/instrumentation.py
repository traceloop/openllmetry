import os
from typing import Collection

from wrapt import wrap_function_wrapper
from opentelemetry.trace import SpanKind, get_tracer, Tracer
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.openai_agents.version import __version__
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry import context as context_api
from .patch import (
    extract_agent_details,
    set_model_settings_span_attributes,
    extract_run_config_details,
    set_prompt_attributes,
    set_response_content_span_attribute,
    set_token_usage_span_attributes,
)


_instruments = ("openai_agents >= 0.0.2",)


class OpenAIAgentsInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI Agents SDK."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        wrap_function_wrapper(
            "agents.run", "Runner._get_new_response", _wrap_agent_run(tracer)
        )

    def _uninstrument(self, **kwargs):
        unwrap("agents.run.Runner", "_get_new_response")


def with_tracer_wrapper(func):

    def _with_tracer(tracer):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


@with_tracer_wrapper
async def _wrap_agent_run(tracer: Tracer, wrapped, instance, args, kwargs):
    agent = args[0]
    run_config = args[7] if len(args) > 7 else None
    prompt_list = args[2] if len(args) > 2 else None
    agent_name = getattr(agent, "name", "agent")
    model_name = getattr(getattr(agent, "model", None), "model",
                         "unknown_model")

    with tracer.start_as_current_span(
        f"openai_agents.{agent_name}",
        kind=SpanKind.INTERNAL,
        attributes={
            SpanAttributes.LLM_SYSTEM: "openai",
            SpanAttributes.LLM_REQUEST_MODEL: model_name,
        },
    ) as span:
        try:
            extract_agent_details(agent, span)
            set_model_settings_span_attributes(agent, span)
            extract_run_config_details(run_config, span)

            response = await wrapped(*args, **kwargs)

            if should_send_prompts():
                if isinstance(prompt_list, list):
                    set_prompt_attributes(span, prompt_list)

                set_response_content_span_attribute(response, span)
                set_token_usage_span_attributes(response, span)

            span.set_status(Status(StatusCode.OK))
            return response

        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value(
        "override_enable_content_tracing"
    )