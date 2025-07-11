"""OpenTelemetry OpenAI Agents instrumentation"""
import os
import time
import json
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
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_COMPLETION,
)
from .utils import set_span_attribute
from agents import FunctionTool, WebSearchTool, FileSearchTool, ComputerTool


_instruments = ("openai-agents >= 0.0.19",)


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
            "AgentRunner._get_new_response",
            _wrap_agent_run(
                tracer,
                duration_histogram,
                token_histogram,
            ),

        )

    def _uninstrument(self, **kwargs):
        unwrap("agents.run.AgentRunner", "_get_new_response")


def with_tracer_wrapper(func):

    def _with_tracer(tracer, duration_histogram, token_histogram):
        async def wrapper(wrapped, instance, args, kwargs):
            return await func(
                tracer,
                duration_histogram,
                token_histogram,
                wrapped,
                instance,
                args,
                kwargs
            )

        return wrapper

    return _with_tracer


@with_tracer_wrapper
async def _wrap_agent_run(
    tracer: Tracer,
    duration_histogram: Histogram,
    token_histogram: Histogram,
    wrapped,
    instance,
    args,
    kwargs,
):
    agent, *_ = args
    run_config = args[7] if len(args) > 7 else None
    prompt_list = args[2] if len(args) > 2 else None
    agent_name = getattr(agent, "name", "agent")
    model_name = get_model_name(agent)

    with tracer.start_as_current_span(
        f"{agent_name}.agent",
        kind=SpanKind.CLIENT,
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
            tools = (
                args[4]
                if len(args) > 4 and isinstance(args[4], list)
                else []
            )
            if tools:
                extract_tool_details(tracer, tools)
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
                response, span, model_name, token_histogram, agent
            )

            span.set_status(Status(StatusCode.OK))
            return response

        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def get_model_name(agent):
    model_attr = getattr(
        getattr(agent, "model", None), "model", "unknown_model"
    )
    if model_attr == "unknown_model":
        model_attr = getattr(agent, "model", None)
        return model_attr
    else:
        return model_attr


def extract_agent_details(test_agent, span):
    if test_agent is None:
        return

    agent = getattr(test_agent, "agent", test_agent)
    if agent is None:
        return

    name = getattr(agent, "name", None)
    instructions = getattr(agent, "instructions", None)
    handoff_description = getattr(agent, "handoff_description", None)
    handoffs = getattr(agent, "handoffs", None)
    if name:
        set_span_attribute(span, "gen_ai.agent.name", name)
    if instructions:
        set_span_attribute(
            span, "gen_ai.agent.description", instructions
        )
    if handoff_description:
        set_span_attribute(
            span, "gen_ai.agent.handoff_description", handoff_description
        )
    if handoffs:
        for idx, h in enumerate(handoffs):
            handoff_info = {
                "name": getattr(h, "name", None),
                "instructions": getattr(h, "instructions", None)
            }
            handoff_json = json.dumps(handoff_info)
            span.set_attribute(f"openai.agent.handoff{idx}", handoff_json)
    attributes = {}
    for key, value in vars(agent).items():
        if key in ("name", "instructions", "handoff_description"):
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


def extract_tool_details(tracer: Tracer, tools):
    for tool in tools:
        if isinstance(tool, FunctionTool):
            tool_name = getattr(tool, "name", "tool")
        elif isinstance(tool, FileSearchTool):
            tool_name = "FileSearchTool"
        elif isinstance(tool, WebSearchTool):
            tool_name = "WebSearchTool"
        elif isinstance(tool, ComputerTool):
            tool_name = "ComputerTool"
        else:
            tool_name = getattr(tool, "name", "unknown_tool")
        with tracer.start_as_current_span(
            f"{tool_name}.tool",
            kind=SpanKind.INTERNAL,
            attributes={
                SpanAttributes.TRACELOOP_SPAN_KIND: (
                    TraceloopSpanKindValues.TOOL.value
                )
            },
        ) as span:
            try:
                if tool_name:
                    if isinstance(tool, FunctionTool):
                        span.set_attribute(
                            f"{GEN_AI_COMPLETION}.tool.name", tool_name
                        )
                        span.set_attribute(
                             f"{GEN_AI_COMPLETION}.tool.type", "FunctionTool"
                        )
                        span.set_attribute(
                            f"{GEN_AI_COMPLETION}.tool.description",
                            tool.description
                        )
                        span.set_attribute(
                             f"{GEN_AI_COMPLETION}.tool.strict_json_schema",
                             tool.strict_json_schema
                        )
                    elif isinstance(tool, FileSearchTool):
                        span.set_attribute(
                            f"{GEN_AI_COMPLETION}.tool.type", "FileSearchTool"
                        )
                        span.set_attribute(
                            f"{GEN_AI_COMPLETION}.tool.vector_store_ids",
                            str(tool.vector_store_ids)
                        )
                        span.set_attribute(
                            f"{GEN_AI_COMPLETION}.tool.max_num_results",
                            tool.max_num_results
                        )
                        span.set_attribute(
                            f"{GEN_AI_COMPLETION}.tool.include_search_results",
                            tool.include_search_results
                        )
                    elif isinstance(tool, WebSearchTool):
                        span.set_attribute(
                            f"{GEN_AI_COMPLETION}.tool.type", "WebSearchTool"
                        )
                        span.set_attribute(
                            f"{GEN_AI_COMPLETION}.tool.search_context_size",
                            tool.search_context_size
                        )
                        if tool.user_location:
                            span.set_attribute(
                                f"{GEN_AI_COMPLETION}.tool.user_location",
                                str(tool.user_location)
                            )
                    elif isinstance(tool, ComputerTool):
                        span.set_attribute(
                            f"{GEN_AI_COMPLETION}.tool.type", "ComputerTool"
                        )
                        span.set_attribute(
                            f"{GEN_AI_COMPLETION}.tool.computer",
                            str(tool.computer)
                        )
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise


def set_prompt_attributes(span, message_history):
    if not message_history:
        return

    for i, msg in enumerate(message_history):
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
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
    response, span, model_name, token_histogram, test_agent
):
    agent = getattr(test_agent, "agent", test_agent)
    if agent is None:
        return

    agent_name = getattr(agent, "name", None)
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
                    "gen_ai.agent.name": agent_name,
                },
            )
            token_histogram.record(
                output_tokens,
                attributes={
                    SpanAttributes.LLM_SYSTEM: "openai",
                    SpanAttributes.LLM_TOKEN_TYPE: "output",
                    SpanAttributes.LLM_RESPONSE_MODEL: model_name,
                    "gen_ai.agent.name": agent_name,
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
