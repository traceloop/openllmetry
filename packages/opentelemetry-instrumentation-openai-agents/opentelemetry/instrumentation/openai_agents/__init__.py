"""OpenTelemetry OpenAI Agents instrumentation"""

import os
import time
import json
import threading
from typing import Collection
from wrapt import wrap_function_wrapper
from opentelemetry.trace import SpanKind, get_tracer, Tracer, set_span_in_context
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry import context
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
from .utils import set_span_attribute, JSONEncoder
from agents import FunctionTool, WebSearchTool, FileSearchTool, ComputerTool


_instruments = ("openai-agents >= 0.0.19",)

_root_span_storage = {}
_instrumented_tools = set()


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
        wrap_function_wrapper(
            "agents.run",
            "AgentRunner._run_single_turn_streamed",
            _wrap_agent_run_streamed(
                tracer,
                duration_histogram,
                token_histogram,
            ),
        )

    def _uninstrument(self, **kwargs):
        unwrap("agents.run.AgentRunner", "_get_new_response")
        unwrap("agents.run.AgentRunner", "_run_single_turn_streamed")
        _instrumented_tools.clear()
        _root_span_storage.clear()


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
                kwargs,
            )

        return wrapper

    return _with_tracer


@with_tracer_wrapper
async def _wrap_agent_run_streamed(
    tracer: Tracer,
    duration_histogram: Histogram,
    token_histogram: Histogram,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Wrapper for _run_single_turn_streamed to handle streaming execution."""
    agent = args[1] if len(args) > 1 else None
    run_config = args[4] if len(args) > 4 else None

    if not agent:
        return await wrapped(*args, **kwargs)

    agent_name = getattr(agent, "name", "agent")
    thread_id = threading.get_ident()

    root_span = _root_span_storage.get(thread_id)

    if root_span:
        ctx = set_span_in_context(root_span, context.get_current())
    else:
        ctx = context.get_current()

    with tracer.start_as_current_span(
        f"{agent_name}.agent",
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: (TraceloopSpanKindValues.AGENT.value),
        },
        context=ctx,
    ) as span:
        try:
            if not root_span:
                _root_span_storage[thread_id] = span

            extract_agent_details(agent, span)
            set_model_settings_span_attributes(agent, span)
            extract_run_config_details(run_config, span)

            try:
                json_args = []
                for arg in args:
                    try:
                        json_args.append(json.loads(json.dumps(arg, cls=JSONEncoder)))
                    except (TypeError, ValueError):
                        json_args.append(str(arg))

                json_kwargs = {}
                for key, value in kwargs.items():
                    try:
                        json_kwargs[key] = json.loads(
                            json.dumps(value, cls=JSONEncoder)
                        )
                    except (TypeError, ValueError):
                        json_kwargs[key] = str(value)

                input_data = {"args": json_args, "kwargs": json_kwargs}
                input_str = json.dumps(input_data)
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, input_str)
            except Exception:
                fallback_data = {
                    "args": [str(arg) for arg in args],
                    "kwargs": {k: str(v) for k, v in kwargs.items()},
                }
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json.dumps(fallback_data))

            tools = getattr(agent, "tools", [])
            if tools:
                extract_tool_details(tracer, tools)

            start_time = time.time()
            result = await wrapped(*args, **kwargs)
            end_time = time.time()

            try:
                output_str = json.dumps(result, cls=JSONEncoder)
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, output_str)
            except Exception:
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json.dumps(str(result)))

            span.set_status(Status(StatusCode.OK))

            if duration_histogram:
                duration = end_time - start_time
                duration_histogram.record(
                    duration,
                    attributes={
                        "gen_ai.agent.name": agent_name,
                    },
                )

            return result

        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


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
    thread_id = threading.get_ident()
    root_span = _root_span_storage.get(thread_id)

    if root_span:
        ctx = set_span_in_context(root_span, context.get_current())
    else:
        ctx = context.get_current()

    with tracer.start_as_current_span(
        f"{agent_name}.agent",
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: (TraceloopSpanKindValues.AGENT.value),
        },
        context=ctx,
    ) as span:
        try:
            if not root_span:
                _root_span_storage[thread_id] = span

            extract_agent_details(agent, span)
            set_model_settings_span_attributes(agent, span)
            extract_run_config_details(run_config, span)

            try:
                json_args = []
                for arg in args:
                    try:
                        json_args.append(json.loads(json.dumps(arg, cls=JSONEncoder)))
                    except (TypeError, ValueError):
                        json_args.append(str(arg))

                json_kwargs = {}
                for key, value in kwargs.items():
                    try:
                        json_kwargs[key] = json.loads(
                            json.dumps(value, cls=JSONEncoder)
                        )
                    except (TypeError, ValueError):
                        json_kwargs[key] = str(value)

                input_data = {"args": json_args, "kwargs": json_kwargs}
                input_str = json.dumps(input_data)
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, input_str)
            except Exception:
                fallback_data = {
                    "args": [str(arg) for arg in args],
                    "kwargs": {k: str(v) for k, v in kwargs.items()},
                }
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json.dumps(fallback_data))

            tools = args[4] if len(args) > 4 and isinstance(args[4], list) else []
            if tools:
                extract_tool_details(tracer, tools)

            start_time = time.time()
            response = await wrapped(*args, **kwargs)

            try:
                output_str = json.dumps(response, cls=JSONEncoder)
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, output_str)
            except Exception:
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json.dumps(str(response)))
            if duration_histogram:
                duration_histogram.record(
                    time.time() - start_time,
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
    model_attr = getattr(getattr(agent, "model", None), "model", "unknown_model")
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
        set_span_attribute(span, "gen_ai.agent.description", instructions)
    if handoff_description:
        set_span_attribute(
            span, "gen_ai.agent.handoff_description", handoff_description
        )
    if handoffs:
        for idx, h in enumerate(handoffs):
            handoff_info = {
                "name": getattr(h, "name", None),
                "instructions": getattr(h, "instructions", None),
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
    """Create spans for hosted tools and wrap FunctionTool execution."""
    thread_id = threading.get_ident()
    root_span = _root_span_storage.get(thread_id)

    for tool in tools:
        if isinstance(tool, FunctionTool):
            tool_id = id(tool)
            if tool_id in _instrumented_tools:
                continue

            _instrumented_tools.add(tool_id)

            original_on_invoke_tool = tool.on_invoke_tool

            def create_wrapped_tool(original_tool, original_func):
                async def wrapped_on_invoke_tool(tool_context, args_json):
                    tool_name = getattr(original_tool, "name", "tool")
                    if root_span:
                        ctx = set_span_in_context(root_span, context.get_current())
                    else:
                        ctx = context.get_current()

                    with tracer.start_as_current_span(
                        f"{tool_name}.tool",
                        kind=SpanKind.INTERNAL,
                        attributes={
                            SpanAttributes.TRACELOOP_SPAN_KIND: (
                                TraceloopSpanKindValues.TOOL.value
                            )
                        },
                        context=ctx,
                    ) as span:
                        try:
                            span.set_attribute(
                                f"{GEN_AI_COMPLETION}.tool.name", tool_name
                            )
                            span.set_attribute(
                                f"{GEN_AI_COMPLETION}.tool.type", "FunctionTool"
                            )
                            span.set_attribute(
                                f"{GEN_AI_COMPLETION}.tool.description",
                                original_tool.description,
                            )
                            span.set_attribute(
                                f"{GEN_AI_COMPLETION}.tool.strict_json_schema",
                                original_tool.strict_json_schema,
                            )
                            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, args_json)
                            result = await original_func(tool_context, args_json)
                            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, str(result))
                            span.set_status(Status(StatusCode.OK))
                            return result
                        except Exception as e:
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                            raise

                return wrapped_on_invoke_tool

            tool.on_invoke_tool = create_wrapped_tool(tool, original_on_invoke_tool)

        elif isinstance(tool, (WebSearchTool, FileSearchTool, ComputerTool)):
            tool_name = type(tool).__name__
            if root_span:
                ctx = set_span_in_context(root_span, context.get_current())
            else:
                ctx = context.get_current()

            span = tracer.start_span(
                f"{tool_name}.tool",
                kind=SpanKind.INTERNAL,
                attributes={
                    SpanAttributes.TRACELOOP_SPAN_KIND: (
                        TraceloopSpanKindValues.TOOL.value
                    )
                },
                context=ctx,
            )

            if isinstance(tool, WebSearchTool):
                span.set_attribute(f"{GEN_AI_COMPLETION}.tool.type", "WebSearchTool")
                span.set_attribute(
                    f"{GEN_AI_COMPLETION}.tool.search_context_size",
                    tool.search_context_size,
                )
                if tool.user_location:
                    span.set_attribute(
                        f"{GEN_AI_COMPLETION}.tool.user_location",
                        str(tool.user_location),
                    )
            elif isinstance(tool, FileSearchTool):
                span.set_attribute(f"{GEN_AI_COMPLETION}.tool.type", "FileSearchTool")
                span.set_attribute(
                    f"{GEN_AI_COMPLETION}.tool.vector_store_ids",
                    str(tool.vector_store_ids),
                )
                if tool.max_num_results:
                    span.set_attribute(
                        f"{GEN_AI_COMPLETION}.tool.max_num_results",
                        tool.max_num_results,
                    )
                span.set_attribute(
                    f"{GEN_AI_COMPLETION}.tool.include_search_results",
                    tool.include_search_results,
                )
            elif isinstance(tool, ComputerTool):
                span.set_attribute(f"{GEN_AI_COMPLETION}.tool.type", "ComputerTool")
                span.set_attribute(
                    f"{GEN_AI_COMPLETION}.tool.computer", str(tool.computer)
                )

            span.set_status(Status(StatusCode.OK))
            span.end()


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

            if hasattr(output_message, "content") and isinstance(
                output_message.content, list
            ):
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
