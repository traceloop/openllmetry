"""OpenTelemetry OpenAI Agents instrumentation"""
import os
import time
from typing import Collection
from wrapt import wrap_function_wrapper
from opentelemetry.trace import SpanKind, get_tracer, Tracer
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.metrics import Histogram, Meter, get_meter
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.openai_agents.version import __version__
from opentelemetry.semconv_ai import (
    SpanAttributes,
    TraceloopSpanKindValues,
    Meters,
)
from .utils import set_span_attribute


_instruments = ("openai-agents >= 0.0.2",)


class OpenAIAgentsInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI Agents SDK."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        if is_metrics_enabled():
            (
                token_histogram,
                duration_histogram,
            ) = _create_metrics(meter)
        else:
            (
                token_histogram,
                duration_histogram,
            ) = (None, None)

        wrap_function_wrapper(
            "agents.run",
            "Runner._get_new_response",
            _wrap_agent_run(
                tracer,
                duration_histogram,
                token_histogram,
            ),
        )

    def _uninstrument(self, **kwargs):
        unwrap("agents.run.Runner", "_get_new_response")


def with_tracer_wrapper(func):

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
async def _wrap_agent_run(
    tracer: Tracer,
    duration_histogram: Histogram,
    token_histogram: Histogram,
    wrapped,
    args,
    kwargs,
):
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
            SpanAttributes.TRACELOOP_SPAN_KIND: (
                TraceloopSpanKindValues.AGENT.value
            )
        },
    ) as span:
        try:

            extract_agent_details(agent, span)
            set_model_settings_span_attributes(agent, span)
            extract_run_config_details(run_config, span)
            start_time = time.time()
            response = await wrapped(*args, **kwargs)
            if duration_histogram:
                duration_histogram.record(
                    time.time() - start_time,
                    attributes={
                        SpanAttributes.LLM_SYSTEM: "openai",
                        SpanAttributes.LLM_RESPONSE_MODEL: model_name,

                    },
                )
            if isinstance(prompt_list, list):
                set_prompt_attributes(span, prompt_list)
            set_response_content_span_attribute(response, span)
            set_token_usage_span_attributes(
                response, span, model_name, token_histogram
            )

            span.set_status(Status(StatusCode.OK))
            return response

        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def extract_agent_details(test_agent, span):
    if test_agent is None:
        return

    agent = getattr(test_agent, "agent", test_agent)
    if agent is None:
        return

    name = getattr(agent, "name", None)
    instructions = getattr(agent, "instructions", None)
    if name:
        set_span_attribute(span, "gen_ai.agent.name", name)
    if instructions:
        set_span_attribute(
            span, "gen_ai.agent.description", instructions
        )

    attributes = {}
    for key, value in vars(agent).items():
        if key in ("name", "instructions"):
            continue

        if value is not None:
            if isinstance(value, (str, int, float, bool)):
                attributes[f"openai.agent.{key}"] = value
            elif isinstance(value, list) and len(value) > 0:
                attributes[f"openai.agent.{key}_count"] = len(value)

    if attributes:
        span.set_attributes(attributes)


def set_model_settings_span_attributes(agent, span):

    if not hasattr(agent, "model_settings") or agent.model_settings is None:
        return

    model_settings = agent.model_settings
    settings_dict = vars(model_settings)

    key_to_span_attr = {
        "max_tokens": SpanAttributes.LLM_REQUEST_MAX_TOKENS,
        "temperature": SpanAttributes.LLM_REQUEST_TEMPERATURE,
        "top_p": SpanAttributes.LLM_REQUEST_TOP_P,
    }

    for key, value in settings_dict.items():
        if value is not None:
            span_attr = key_to_span_attr.get(key, f"openai.agent.model.{key}")
            span.set_attribute(span_attr, value)


def extract_run_config_details(run_config, span):
    if run_config is None:
        return

    config_dict = vars(run_config)
    attributes = {}

    for key, value in config_dict.items():

        if value is not None and isinstance(value, (str, int, float, bool)):
            attributes[f"openai.agent.{key}"] = value
        elif isinstance(value, list) and len(value) != 0:
            attributes[f"openai.agent.{key}_count"] = len(value)

    if attributes:
        span.set_attributes(attributes)


def set_prompt_attributes(span, message_history):
    if not message_history:
        return

    for i, msg in enumerate(message_history):
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", None)
            set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.{i}.role",
                role,
            )
            set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                content,
            )


def set_response_content_span_attribute(response, span):
    if hasattr(response, "output") and isinstance(response.output, list):
        roles = []
        types = []
        contents = []

        for output_message in response.output:
            role = getattr(output_message, "role", None)
            msg_type = getattr(output_message, "type", None)

            if role:
                roles.append(role)
            if msg_type:
                types.append(msg_type)

            if hasattr(output_message, "content") and \
                    isinstance(output_message.content, list):
                for content_item in output_message.content:
                    if hasattr(content_item, "text"):
                        contents.append(content_item.text)

        if roles:
            set_span_attribute(
                span,
                f"{SpanAttributes.LLM_COMPLETIONS}.roles",
                roles,
            )
        if types:
            set_span_attribute(
                span,
                f"{SpanAttributes.LLM_COMPLETIONS}.types",
                types,
            )
        if contents:
            set_span_attribute(
                span, f"{SpanAttributes.LLM_COMPLETIONS}.contents", contents
            )


def set_token_usage_span_attributes(
    response, span, model_name, token_histogram
):
    if hasattr(response, "usage"):
        usage = response.usage
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

        if input_tokens is not None:
            set_span_attribute(
                span,
                SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
                input_tokens,
            )
        if output_tokens is not None:
            set_span_attribute(
                span,
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                output_tokens,
            )
        if total_tokens is not None:
            set_span_attribute(
                span,
                SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
                total_tokens,
            )
        if token_histogram:
            token_histogram.record(
                input_tokens,
                attributes={
                    SpanAttributes.LLM_SYSTEM: "openai",
                    SpanAttributes.LLM_TOKEN_TYPE: "input",
                    SpanAttributes.LLM_RESPONSE_MODEL: model_name,
                },
            )
            token_histogram.record(
                output_tokens,
                attributes={
                    SpanAttributes.LLM_SYSTEM: "openai",
                    SpanAttributes.LLM_TOKEN_TYPE: "output",
                    SpanAttributes.LLM_RESPONSE_MODEL: model_name,
                },
            )


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
