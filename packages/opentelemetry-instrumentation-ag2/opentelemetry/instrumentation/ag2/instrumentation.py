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
from opentelemetry.instrumentation.ag2.version import __version__
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues, Meters
from .ag2_span_attributes import AG2SpanAttributes

_instruments = ("ag2 >= 0.11.0",)


class AG2Instrumentor(BaseInstrumentor):

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

        # Conversation spans (sync + async)
        wrap_function_wrapper("autogen.agentchat.conversable_agent", "ConversableAgent.initiate_chat",
                              wrap_initiate_chat(tracer, duration_histogram, token_histogram))
        wrap_function_wrapper("autogen.agentchat.conversable_agent", "ConversableAgent.a_initiate_chat",
                              wrap_a_initiate_chat(tracer, duration_histogram, token_histogram))

        # Agent invocation spans (sync + async)
        wrap_function_wrapper("autogen.agentchat.conversable_agent", "ConversableAgent.generate_reply",
                              wrap_generate_reply(tracer, duration_histogram, token_histogram))
        wrap_function_wrapper("autogen.agentchat.conversable_agent", "ConversableAgent.a_generate_reply",
                              wrap_a_generate_reply(tracer, duration_histogram, token_histogram))

        # Tool execution spans (sync + async)
        wrap_function_wrapper("autogen.agentchat.conversable_agent", "ConversableAgent.execute_function",
                              wrap_execute_function(tracer, duration_histogram, token_histogram))
        wrap_function_wrapper("autogen.agentchat.conversable_agent", "ConversableAgent.a_execute_function",
                              wrap_a_execute_function(tracer, duration_histogram, token_histogram))

        # Group chat conversation spans (sync + async)
        wrap_function_wrapper("autogen.agentchat.groupchat", "GroupChatManager.run_chat",
                              wrap_run_chat(tracer, duration_histogram, token_histogram))
        wrap_function_wrapper("autogen.agentchat.groupchat", "GroupChatManager.a_run_chat",
                              wrap_a_run_chat(tracer, duration_histogram, token_histogram))

    def _uninstrument(self, **kwargs):
        unwrap("autogen.agentchat.conversable_agent.ConversableAgent", "initiate_chat")
        unwrap("autogen.agentchat.conversable_agent.ConversableAgent", "a_initiate_chat")
        unwrap("autogen.agentchat.conversable_agent.ConversableAgent", "generate_reply")
        unwrap("autogen.agentchat.conversable_agent.ConversableAgent", "a_generate_reply")
        unwrap("autogen.agentchat.conversable_agent.ConversableAgent", "execute_function")
        unwrap("autogen.agentchat.conversable_agent.ConversableAgent", "a_execute_function")
        unwrap("autogen.agentchat.groupchat.GroupChatManager", "run_chat")
        unwrap("autogen.agentchat.groupchat.GroupChatManager", "a_run_chat")


def with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, duration_histogram, token_histogram):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs)
        return wrapper
    return _with_tracer


def with_tracer_wrapper_async(func):
    """Helper for providing tracer for async wrapper functions."""

    def _with_tracer(tracer, duration_histogram, token_histogram):
        async def wrapper(wrapped, instance, args, kwargs):
            return await func(tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs)
        return wrapper
    return _with_tracer


def _should_send_prompts():
    return (os.getenv("TRACELOOP_TRACE_CONTENT") or "true").lower() == "true"


def _get_provider_name(agent):
    """Extract LLM provider name from agent's llm_config."""
    llm_config = getattr(agent, "llm_config", None)
    if not llm_config or not isinstance(llm_config, dict):
        return None
    config_list = llm_config.get("config_list", [])
    if config_list:
        return config_list[0].get("api_type", None)
    return None


def _get_model_name(agent):
    """Extract model name from agent's llm_config."""
    llm_config = getattr(agent, "llm_config", None)
    if not llm_config or not isinstance(llm_config, dict):
        return None
    config_list = llm_config.get("config_list", [])
    if config_list:
        return config_list[0].get("model", None)
    model = llm_config.get("model", None)
    return model


def _set_conversation_result_attributes(span, result, token_histogram):
    """Set span attributes from a ChatResult object."""
    if not result:
        return
    if hasattr(result, "chat_id"):
        span.set_attribute("gen_ai.conversation.id", str(result.chat_id))
    if hasattr(result, "chat_history"):
        span.set_attribute("gen_ai.conversation.turns", len(result.chat_history))
        if _should_send_prompts() and result.chat_history:
            span.set_attribute("gen_ai.output.messages", json.dumps(result.chat_history))
    if hasattr(result, "cost") and result.cost:
        usage_incl = result.cost.get("usage_including_cached_inference", {})
        if usage_incl:
            total_cost = usage_incl.get("total_cost", 0)
            span.set_attribute("gen_ai.usage.cost", total_cost)
            for model_name, model_usage in usage_incl.items():
                if model_name == "total_cost" or not isinstance(model_usage, dict):
                    continue
                input_tokens = model_usage.get("prompt_tokens", 0)
                output_tokens = model_usage.get("completion_tokens", 0)
                if input_tokens or output_tokens:
                    span.set_attribute("gen_ai.response.model", model_name)
                    span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                    span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
                    if token_histogram:
                        token_histogram.record(
                            input_tokens,
                            attributes={
                                GenAIAttributes.GEN_AI_SYSTEM: "ag2",
                                GenAIAttributes.GEN_AI_TOKEN_TYPE: "input",
                                GenAIAttributes.GEN_AI_RESPONSE_MODEL: model_name,
                            }
                        )
                        token_histogram.record(
                            output_tokens,
                            attributes={
                                GenAIAttributes.GEN_AI_SYSTEM: "ag2",
                                GenAIAttributes.GEN_AI_TOKEN_TYPE: "output",
                                GenAIAttributes.GEN_AI_RESPONSE_MODEL: model_name,
                            }
                        )
                    break  # Use first model's usage


def _set_initiate_chat_span_attrs(span, instance, args, kwargs):
    """Set common attributes for initiate_chat spans."""
    agent_name = instance.name if hasattr(instance, "name") else "agent"
    span.set_attribute("ag2.span.type", "conversation")
    span.set_attribute("gen_ai.operation.name", "conversation")
    span.set_attribute("gen_ai.agent.name", agent_name)

    # Set provider and model from recipient (first positional arg)
    if args:
        recipient = args[0]
        provider = _get_provider_name(recipient)
        if provider:
            span.set_attribute("gen_ai.provider.name", provider)
        model = _get_model_name(recipient)
        if model:
            span.set_attribute(GenAIAttributes.GEN_AI_REQUEST_MODEL, model)

    max_turns = kwargs.get("max_turns")
    if max_turns:
        span.set_attribute("gen_ai.conversation.max_turns", max_turns)

    if _should_send_prompts():
        message = kwargs.get("message") or (args[1] if len(args) > 1 else None)
        if message:
            if isinstance(message, str):
                input_msg = [{"role": "user", "content": message}]
            elif isinstance(message, dict):
                input_msg = [{"role": message.get("role", "user"), **message}]
            else:
                input_msg = None
            if input_msg:
                span.set_attribute("gen_ai.input.messages", json.dumps(input_msg))

    AG2SpanAttributes(span=span, instance=instance)


@with_tracer_wrapper
def wrap_initiate_chat(tracer: Tracer, duration_histogram: Histogram, token_histogram: Histogram,
                       wrapped, instance, args, kwargs):
    agent_name = instance.name if hasattr(instance, "name") else "agent"
    with tracer.start_as_current_span(
        f"conversation {agent_name}",
        kind=SpanKind.INTERNAL,
        attributes={
            GenAIAttributes.GEN_AI_SYSTEM: "ag2",
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.WORKFLOW.value,
        }
    ) as span:
        start_time = time.time()
        try:
            _set_initiate_chat_span_attrs(span, instance, args, kwargs)
            result = wrapped(*args, **kwargs)
            _set_conversation_result_attributes(span, result, token_histogram)
            if duration_histogram:
                duration = time.time() - start_time
                duration_histogram.record(
                    duration,
                    attributes={GenAIAttributes.GEN_AI_SYSTEM: "ag2"},
                )
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


@with_tracer_wrapper_async
async def wrap_a_initiate_chat(tracer: Tracer, duration_histogram: Histogram, token_histogram: Histogram,
                               wrapped, instance, args, kwargs):
    agent_name = instance.name if hasattr(instance, "name") else "agent"
    with tracer.start_as_current_span(
        f"conversation {agent_name}",
        kind=SpanKind.INTERNAL,
        attributes={
            GenAIAttributes.GEN_AI_SYSTEM: "ag2",
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.WORKFLOW.value,
        }
    ) as span:
        start_time = time.time()
        try:
            _set_initiate_chat_span_attrs(span, instance, args, kwargs)
            result = await wrapped(*args, **kwargs)
            _set_conversation_result_attributes(span, result, token_histogram)
            if duration_histogram:
                duration = time.time() - start_time
                duration_histogram.record(
                    duration,
                    attributes={GenAIAttributes.GEN_AI_SYSTEM: "ag2"},
                )
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


def _set_generate_reply_span_attrs(span, instance, args, kwargs):
    """Set common attributes for generate_reply spans."""
    agent_name = instance.name if hasattr(instance, "name") else "agent"
    span.set_attribute("ag2.span.type", "agent")
    span.set_attribute("gen_ai.operation.name", "invoke_agent")
    span.set_attribute("gen_ai.agent.name", agent_name)

    provider = _get_provider_name(instance)
    if provider:
        span.set_attribute("gen_ai.provider.name", provider)
    model = _get_model_name(instance)
    if model:
        span.set_attribute(GenAIAttributes.GEN_AI_REQUEST_MODEL, model)

    if _should_send_prompts():
        messages = kwargs.get("messages") or (args[0] if args else None)
        if messages:
            span.set_attribute("gen_ai.input.messages", json.dumps(messages))

    AG2SpanAttributes(span=span, instance=instance)


@with_tracer_wrapper
def wrap_generate_reply(tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs):
    agent_name = instance.name if hasattr(instance, "name") else "agent"
    with tracer.start_as_current_span(
        f"invoke_agent {agent_name}",
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.AGENT.value,
        }
    ) as span:
        start_time = time.time()
        try:
            _set_generate_reply_span_attrs(span, instance, args, kwargs)
            result = wrapped(*args, **kwargs)
            if _should_send_prompts() and result is not None:
                if isinstance(result, str):
                    output = [{"role": "assistant", "content": result}]
                elif isinstance(result, dict):
                    output = [result]
                else:
                    output = [{"role": "assistant", "content": str(result)}]
                span.set_attribute("gen_ai.output.messages", json.dumps(output))
            if duration_histogram:
                duration = time.time() - start_time
                duration_histogram.record(
                    duration,
                    attributes={GenAIAttributes.GEN_AI_SYSTEM: "ag2"},
                )
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


@with_tracer_wrapper_async
async def wrap_a_generate_reply(tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs):
    agent_name = instance.name if hasattr(instance, "name") else "agent"
    with tracer.start_as_current_span(
        f"invoke_agent {agent_name}",
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.AGENT.value,
        }
    ) as span:
        start_time = time.time()
        try:
            _set_generate_reply_span_attrs(span, instance, args, kwargs)
            result = await wrapped(*args, **kwargs)
            if _should_send_prompts() and result is not None:
                if isinstance(result, str):
                    output = [{"role": "assistant", "content": result}]
                elif isinstance(result, dict):
                    output = [result]
                else:
                    output = [{"role": "assistant", "content": str(result)}]
                span.set_attribute("gen_ai.output.messages", json.dumps(output))
            if duration_histogram:
                duration = time.time() - start_time
                duration_histogram.record(
                    duration,
                    attributes={GenAIAttributes.GEN_AI_SYSTEM: "ag2"},
                )
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


def _set_execute_function_span_attrs(span, func_call):
    """Set common attributes for execute_function spans."""
    func_name = func_call.get("name", "unknown") if isinstance(func_call, dict) else "unknown"
    span.set_attribute("ag2.span.type", "tool")
    span.set_attribute("gen_ai.operation.name", "execute_tool")
    span.set_attribute("gen_ai.tool.name", func_name)
    span.set_attribute("gen_ai.tool.type", "function")

    if _should_send_prompts() and isinstance(func_call, dict):
        arguments = func_call.get("arguments", "")
        if arguments:
            if isinstance(arguments, str):
                span.set_attribute("gen_ai.tool.call.arguments", arguments)
            else:
                span.set_attribute("gen_ai.tool.call.arguments", json.dumps(arguments))
    return func_name


@with_tracer_wrapper
def wrap_execute_function(tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs):
    func_call = kwargs.get("func_call") or (args[0] if args else {})
    func_name = func_call.get("name", "unknown") if isinstance(func_call, dict) else "unknown"
    with tracer.start_as_current_span(
        f"execute_tool {func_name}",
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.TOOL.value,
        }
    ) as span:
        start_time = time.time()
        try:
            _set_execute_function_span_attrs(span, func_call)
            is_success, result = wrapped(*args, **kwargs)
            if not is_success:
                span.set_attribute("error.type", "ExecutionError")
            elif _should_send_prompts():
                content = result.get("content", "") if isinstance(result, dict) else str(result)
                span.set_attribute("gen_ai.tool.call.result", str(content))
            if duration_histogram:
                duration = time.time() - start_time
                duration_histogram.record(
                    duration,
                    attributes={GenAIAttributes.GEN_AI_SYSTEM: "ag2"},
                )
            span.set_status(Status(StatusCode.OK))
            return is_success, result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


@with_tracer_wrapper_async
async def wrap_a_execute_function(tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs):
    func_call = kwargs.get("func_call") or (args[0] if args else {})
    func_name = func_call.get("name", "unknown") if isinstance(func_call, dict) else "unknown"
    with tracer.start_as_current_span(
        f"execute_tool {func_name}",
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.TOOL.value,
        }
    ) as span:
        start_time = time.time()
        try:
            _set_execute_function_span_attrs(span, func_call)
            is_success, result = await wrapped(*args, **kwargs)
            if not is_success:
                span.set_attribute("error.type", "ExecutionError")
            elif _should_send_prompts():
                content = result.get("content", "") if isinstance(result, dict) else str(result)
                span.set_attribute("gen_ai.tool.call.result", str(content))
            if duration_histogram:
                duration = time.time() - start_time
                duration_histogram.record(
                    duration,
                    attributes={GenAIAttributes.GEN_AI_SYSTEM: "ag2"},
                )
            span.set_status(Status(StatusCode.OK))
            return is_success, result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


def _set_run_chat_span_attrs(span, instance, args, kwargs):
    """Set common attributes for run_chat spans."""
    agent_name = instance.name if hasattr(instance, "name") else "chat_manager"
    span.set_attribute("ag2.span.type", "conversation")
    span.set_attribute("gen_ai.operation.name", "conversation")
    span.set_attribute("gen_ai.agent.name", agent_name)

    if _should_send_prompts():
        messages = kwargs.get("messages") or (args[0] if args else None)
        if messages:
            span.set_attribute("gen_ai.input.messages", json.dumps(messages))

    AG2SpanAttributes(span=span, instance=instance)


@with_tracer_wrapper
def wrap_run_chat(tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs):
    agent_name = instance.name if hasattr(instance, "name") else "chat_manager"
    with tracer.start_as_current_span(
        f"conversation {agent_name}",
        kind=SpanKind.INTERNAL,
        attributes={
            GenAIAttributes.GEN_AI_SYSTEM: "ag2",
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.WORKFLOW.value,
        }
    ) as span:
        start_time = time.time()
        try:
            _set_run_chat_span_attrs(span, instance, args, kwargs)
            result = wrapped(*args, **kwargs)
            # Capture output from groupchat config
            config = kwargs.get("config") or (args[2] if len(args) > 2 else None)
            if _should_send_prompts() and config and hasattr(config, "messages") and config.messages:
                span.set_attribute("gen_ai.output.messages", json.dumps(config.messages))
            if duration_histogram:
                duration = time.time() - start_time
                duration_histogram.record(
                    duration,
                    attributes={GenAIAttributes.GEN_AI_SYSTEM: "ag2"},
                )
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


@with_tracer_wrapper_async
async def wrap_a_run_chat(tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs):
    agent_name = instance.name if hasattr(instance, "name") else "chat_manager"
    with tracer.start_as_current_span(
        f"conversation {agent_name}",
        kind=SpanKind.INTERNAL,
        attributes={
            GenAIAttributes.GEN_AI_SYSTEM: "ag2",
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.WORKFLOW.value,
        }
    ) as span:
        start_time = time.time()
        try:
            _set_run_chat_span_attrs(span, instance, args, kwargs)
            result = await wrapped(*args, **kwargs)
            config = kwargs.get("config") or (args[2] if len(args) > 2 else None)
            if _should_send_prompts() and config and hasattr(config, "messages") and config.messages:
                span.set_attribute("gen_ai.output.messages", json.dumps(config.messages))
            if duration_histogram:
                duration = time.time() - start_time
                duration_histogram.record(
                    duration,
                    attributes={GenAIAttributes.GEN_AI_SYSTEM: "ag2"},
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
