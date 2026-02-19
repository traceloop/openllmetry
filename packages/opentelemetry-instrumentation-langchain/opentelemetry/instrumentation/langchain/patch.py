"""Patching utilities for LangGraph instrumentation."""

import hashlib
import json
import logging
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.trace import Tracer, Span, SpanKind, Status, StatusCode
from opentelemetry.semconv_ai import GenAIOperationName, SpanAttributes
from typing import Any

logger = logging.getLogger(__name__)

# Import ContextVar for reading current node (set by callback_handler)
# Importing at runtime to avoid circular import issues
def _get_current_node_contextvar():
    """Lazy import to avoid circular dependency."""
    from opentelemetry.instrumentation.langchain.callback_handler import _langgraph_current_node
    return _langgraph_current_node


def _generate_agent_id(name: str, instance: Any) -> str:
    """Generate a unique agent ID by hashing the instance memory address."""
    instance_id = str(id(instance))
    hash_input = f"{name}_{instance_id}"
    hashed = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    return f"{name}_{hashed}"

# Context key for marking LangGraph flow
LANGGRAPH_FLOW_KEY = "langgraph_flow"
# Context key for storing the graph span as parent for callback-created spans
LANGGRAPH_GRAPH_SPAN_KEY = "langgraph_graph_span"
# Context key for tracking if first child of graph span is pending (mutable list [bool])
LANGGRAPH_FIRST_CHILD_PENDING_KEY = "langgraph_first_child_pending"


def _set_graph_span_attributes(
    graph_span: Span,
    instance: Any,
    graph_name: str,
    kwargs: dict,
    args: tuple
) -> None:
    """
    Set common GenAI attributes on graph span.

    This helper function consolidates attribute setting to avoid duplication
    between sync and async wrappers.

    Args:
        graph_span: The span to set attributes on
        instance: The graph instance
        graph_name: Name of the graph
        kwargs: Keyword arguments passed to the graph invocation
        args: Positional arguments passed to the graph invocation
    """
    from opentelemetry.instrumentation.langchain.langgraph_utils import extract_graph_structure

    # Set GenAI semantic convention attributes
    graph_span.set_attribute(SpanAttributes.GEN_AI_PROVIDER_NAME, "langgraph")
    graph_span.set_attribute(
        SpanAttributes.GEN_AI_OPERATION_NAME, GenAIOperationName.INVOKE_AGENT.value
    )
    graph_span.set_attribute(SpanAttributes.GEN_AI_AGENT_NAME, graph_name)
    graph_span.set_attribute(
        SpanAttributes.GEN_AI_AGENT_ID, _generate_agent_id(graph_name, instance)
    )

    # Extract conversation ID from config
    config = kwargs.get('config') or (args[1] if len(args) > 1 else None)
    if config and isinstance(config, dict):
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        if thread_id:
            graph_span.set_attribute(SpanAttributes.GEN_AI_CONVERSATION_ID, str(thread_id))

    # Extract workflow structure (best-effort with debug logging)
    try:
        nodes, edges = extract_graph_structure(instance)
        if nodes:
            graph_span.set_attribute(SpanAttributes.GEN_AI_WORKFLOW_NODES, nodes)
        if edges:
            graph_span.set_attribute(SpanAttributes.GEN_AI_WORKFLOW_EDGES, edges)
    except Exception as e:
        logger.debug("Failed to extract LangGraph workflow structure: %s", e)


def _get_graph_name(instance, args, kwargs) -> str:
    """
    Get the graph name from available sources in order of priority:
    1. config['run_name'] (from args[1] or kwargs['config'])
    2. instance.get_name() method (matches LangGraph's behavior)
    3. Default to "LangGraph"

    Note: stream/astream signature is (self, input, config=None, *, ...)
    so config can be args[1] (positional) or kwargs['config'] (keyword).
    """
    # Config can be in args[1] (positional) or kwargs['config'] (keyword)
    config = None
    if len(args) > 1:
        config = args[1]
    if config is None:
        config = kwargs.get('config')

    # Try run_name from config first (config could be RunnableConfig object, not dict)
    if config and isinstance(config, dict):
        run_name = config.get('run_name')
        if run_name:
            return run_name

    # Fallback to instance.get_name() to match LangGraph behavior
    if hasattr(instance, 'get_name'):
        return instance.get_name()

    # Default
    return "LangGraph"


def create_graph_invocation_wrapper(tracer: Tracer, is_async: bool = False):
    """
    Factory to create wrappers for graph invocation methods.

    Args:
        tracer: OpenTelemetry tracer instance
        is_async: Whether to create an async wrapper

    Returns:
        Wrapper function for sync or async graph invocation
    """
    def wrapper(wrapped, instance, args, kwargs):
        """Wrapper for Pregel.stream - yields from the generator while managing span lifecycle."""
        graph_name = _get_graph_name(instance, args, kwargs)

        # Set LangGraph flow context before creating spans
        langgraph_ctx = context_api.attach(
            context_api.set_value(LANGGRAPH_FLOW_KEY, graph_name)
        )

        # Create graph span with GenAI convention naming: invoke_agent {agent_name}
        graph_span = tracer.start_span(f"invoke_agent {graph_name}")

        # Set all graph span attributes using helper function
        _set_graph_span_attributes(graph_span, instance, graph_name, kwargs, args)

        # Attach span to context for parent-child linking
        ctx_with_span = trace.set_span_in_context(graph_span)
        ctx_with_span = context_api.set_value(LANGGRAPH_GRAPH_SPAN_KEY, graph_span, ctx_with_span)
        ctx_with_span = context_api.set_value(LANGGRAPH_FIRST_CHILD_PENDING_KEY, [True], ctx_with_span)
        graph_span_ctx = context_api.attach(ctx_with_span)

        try:
            for item in wrapped(*args, **kwargs):
                yield item
        except BaseException as e:
            graph_span.set_status(Status(StatusCode.ERROR, str(e)))
            graph_span.record_exception(e)
            raise
        finally:
            graph_span.end()
            context_api.detach(graph_span_ctx)
            context_api.detach(langgraph_ctx)

    async def async_wrapper(wrapped, instance, args, kwargs):
        """Wrapper for Pregel.astream - yields from the async generator while managing span lifecycle."""
        graph_name = _get_graph_name(instance, args, kwargs)

        # Set LangGraph flow context before creating spans
        langgraph_ctx = context_api.attach(
            context_api.set_value(LANGGRAPH_FLOW_KEY, graph_name)
        )

        # Create graph span with GenAI convention naming: invoke_agent {agent_name}
        graph_span = tracer.start_span(f"invoke_agent {graph_name}")

        # Set all graph span attributes using helper function
        _set_graph_span_attributes(graph_span, instance, graph_name, kwargs, args)

        # Attach span to context for parent-child linking
        ctx_with_span = trace.set_span_in_context(graph_span)
        ctx_with_span = context_api.set_value(LANGGRAPH_GRAPH_SPAN_KEY, graph_span, ctx_with_span)
        ctx_with_span = context_api.set_value(LANGGRAPH_FIRST_CHILD_PENDING_KEY, [True], ctx_with_span)
        graph_span_ctx = context_api.attach(ctx_with_span)

        try:
            async for item in wrapped(*args, **kwargs):
                yield item
        except BaseException as e:
            graph_span.set_status(Status(StatusCode.ERROR, str(e)))
            graph_span.record_exception(e)
            raise
        finally:
            graph_span.end()
            context_api.detach(graph_span_ctx)
            context_api.detach(langgraph_ctx)

    return async_wrapper if is_async else wrapper


def create_command_init_wrapper(tracer: Tracer):
    """
    Wrapper for Command.__init__ to capture command creation.

    Creates a span when a Command object is created, capturing only:
    - Source node (from context)
    - Destination node(s) from goto parameter

    Args:
        tracer: OpenTelemetry tracer instance

    Returns:
        Wrapper function for Command.__init__
    """
    def wrapper(wrapped, instance, args, kwargs):
        # Call original __init__ first
        result = wrapped(*args, **kwargs)

        # Only create span if goto is specified (indicates routing)
        if instance.goto:
            # Get source node from ContextVar (set by callback_handler)
            source_node = _get_current_node_contextvar().get()

            # Extract goto destination(s)
            goto_destinations = _extract_goto_destinations(instance.goto)

            # Create span only if we have both source and destination
            if source_node and isinstance(source_node, str) and goto_destinations:
                # Format span name as "goto {target}" or "goto {target1, target2}" for multiple
                if len(goto_destinations) == 1:
                    target_str = goto_destinations[0]
                else:
                    target_str = ", ".join(goto_destinations)
                span_name = f"goto {target_str}"

                with tracer.start_as_current_span(
                    span_name,
                    kind=SpanKind.INTERNAL
                ) as span:
                    # Set GenAI operation name
                    span.set_attribute(SpanAttributes.GEN_AI_OPERATION_NAME, "goto")

                    span.set_attribute(
                        SpanAttributes.LANGGRAPH_COMMAND_SOURCE_NODE, source_node
                    )

                    if len(goto_destinations) == 1:
                        span.set_attribute(
                            SpanAttributes.LANGGRAPH_COMMAND_GOTO_NODE, goto_destinations[0]
                        )
                    else:
                        span.set_attribute(
                            SpanAttributes.LANGGRAPH_COMMAND_GOTO_NODES, json.dumps(goto_destinations)
                        )

        return result

    return wrapper


def _extract_goto_destinations(goto: Any) -> list[str]:
    """
    Extract destination node names from goto parameter.

    Args:
        goto: Can be string, Send, or sequence of strings/Sends

    Returns:
        List of destination node names
    """
    try:
        from langgraph.types import Send
    except ImportError:
        # If Send is not available, just handle strings
        Send = type(None)

    destinations = []

    if isinstance(goto, str):
        destinations.append(goto)
    elif Send is not type(None) and isinstance(goto, Send):
        destinations.append(goto.node)
    elif isinstance(goto, (list, tuple)):
        for item in goto:
            if isinstance(item, str):
                destinations.append(item)
            elif Send is not type(None) and isinstance(item, Send):
                destinations.append(item.node)

    return destinations


def _set_middleware_span_attributes(
    span: Span,
    middleware_name: str,
    hook_name: str
) -> None:
    """
    Set common GenAI attributes on middleware span.

    This helper function consolidates attribute setting to avoid duplication
    between sync and async middleware wrappers.

    Args:
        span: The span to set attributes on
        middleware_name: Name of the middleware class
        hook_name: Name of the hook being executed
    """
    span.set_attribute(
        SpanAttributes.GEN_AI_OPERATION_NAME,
        GenAIOperationName.EXECUTE_TASK.value,
    )
    span.set_attribute(SpanAttributes.GEN_AI_TASK_KIND, middleware_name)
    span.set_attribute(
        SpanAttributes.GEN_AI_TASK_NAME, f"{middleware_name}.{hook_name}"
    )
    span.set_attribute(SpanAttributes.GEN_AI_PROVIDER_NAME, "langchain")


def create_middleware_hook_wrapper(tracer: Tracer, hook_name: str):
    """
    Wrapper for AgentMiddleware hook methods (before_model, after_model, etc.)

    Creates a span when a middleware hook is called, capturing:
    - Middleware class name as gen_ai.task.kind
    - Hook name as part of gen_ai.task.name

    Args:
        tracer: OpenTelemetry tracer instance
        hook_name: Name of the hook being wrapped (e.g., "before_model")

    Returns:
        Wrapper function for the middleware hook
    """
    def wrapper(wrapped, instance, args, kwargs):
        middleware_name = instance.__class__.__name__
        span_name = f"execute_task {middleware_name}.{hook_name}"

        with tracer.start_as_current_span(span_name, kind=SpanKind.INTERNAL) as span:
            _set_middleware_span_attributes(span, middleware_name, hook_name)

            try:
                result = wrapped(*args, **kwargs)
                span.set_attribute(SpanAttributes.GEN_AI_TASK_STATUS, "success")
                return result
            except Exception as e:
                span.set_attribute(SpanAttributes.GEN_AI_TASK_STATUS, "failure")
                span.record_exception(e)
                raise

    return wrapper


def create_async_middleware_hook_wrapper(tracer: Tracer, hook_name: str):
    """
    Async wrapper for AgentMiddleware hook methods (abefore_model, aafter_model, etc.)

    Args:
        tracer: OpenTelemetry tracer instance
        hook_name: Name of the hook being wrapped (e.g., "abefore_model")

    Returns:
        Async wrapper function for the middleware hook
    """
    async def async_wrapper(wrapped, instance, args, kwargs):
        middleware_name = instance.__class__.__name__
        span_name = f"execute_task {middleware_name}.{hook_name}"

        with tracer.start_as_current_span(span_name, kind=SpanKind.INTERNAL) as span:
            _set_middleware_span_attributes(span, middleware_name, hook_name)

            try:
                result = await wrapped(*args, **kwargs)
                span.set_attribute(SpanAttributes.GEN_AI_TASK_STATUS, "success")
                return result
            except Exception as e:
                span.set_attribute(SpanAttributes.GEN_AI_TASK_STATUS, "failure")
                span.record_exception(e)
                raise

    return async_wrapper


def _extract_tool_definition(tool: Any) -> dict | None:
    """
    Extract tool definition in OpenAI function format.

    Returns a dict with type, name, description, and parameters.
    """
    tool_def = {"type": "function"}

    # Extract name
    if hasattr(tool, 'name'):
        tool_def["name"] = tool.name
    elif isinstance(tool, dict) and 'name' in tool:
        tool_def["name"] = tool['name']
    elif hasattr(tool, '__name__'):
        tool_def["name"] = tool.__name__
    else:
        return None

    # Extract description
    if hasattr(tool, 'description'):
        tool_def["description"] = tool.description
    elif isinstance(tool, dict) and 'description' in tool:
        tool_def["description"] = tool['description']
    elif hasattr(tool, '__doc__') and tool.__doc__:
        tool_def["description"] = tool.__doc__

    # Extract parameters schema
    parameters = None
    if hasattr(tool, 'args_schema') and tool.args_schema:
        # LangChain tools with Pydantic schema
        try:
            if hasattr(tool.args_schema, 'model_json_schema'):
                parameters = tool.args_schema.model_json_schema()
            elif hasattr(tool.args_schema, 'schema'):
                parameters = tool.args_schema.schema()
        except Exception:
            pass
    elif isinstance(tool, dict) and 'parameters' in tool:
        parameters = tool['parameters']

    if parameters:
        tool_def["parameters"] = parameters

    return tool_def


def create_agent_wrapper(tracer: Tracer, provider_name: str = "langchain"):
    """
    Wrapper for create_agent factory functions.

    Captures agent creation with GenAI semantic convention attributes.

    Args:
        tracer: OpenTelemetry tracer instance
        provider_name: The provider name to use (e.g., "langgraph" or "langchain")

    Returns:
        Wrapper function for agent factory
    """
    def wrapper(wrapped, _instance, args, kwargs):
        # Extract agent name from kwargs or use function name
        agent_name = kwargs.get("name")
        if not agent_name:
            # Use the wrapped function's name as fallback
            agent_name = getattr(wrapped, '__name__', 'agent')
            # Clean up the name (e.g., "create_react_agent" -> "react_agent")
            if agent_name.startswith('create_'):
                agent_name = agent_name[7:]

        span_name = f"create_agent {agent_name}"

        with tracer.start_as_current_span(span_name, kind=SpanKind.INTERNAL) as span:
            span.set_attribute(
                SpanAttributes.GEN_AI_OPERATION_NAME,
                GenAIOperationName.CREATE_AGENT.value,
            )
            span.set_attribute(SpanAttributes.GEN_AI_PROVIDER_NAME, provider_name)
            span.set_attribute(SpanAttributes.GEN_AI_AGENT_NAME, agent_name)

            # Extract system instructions from prompt/system_prompt parameter
            # LangGraph uses "prompt", LangChain uses "system_prompt"
            system_instructions = kwargs.get("prompt") or kwargs.get("system_prompt")
            if system_instructions:
                if isinstance(system_instructions, str):
                    span.set_attribute(SpanAttributes.GEN_AI_SYSTEM_INSTRUCTIONS, system_instructions)
                elif hasattr(system_instructions, 'content'):
                    # SystemMessage or similar object with content attribute
                    span.set_attribute(
                        SpanAttributes.GEN_AI_SYSTEM_INSTRUCTIONS, str(system_instructions.content)
                    )

            # Extract tool definitions in OpenAI function format
            # Tools can be in args[1] (positional) or kwargs
            tools = kwargs.get("tools")
            if tools is None and len(args) > 1:
                tools = args[1]
            if tools:
                tool_definitions = []
                for tool in tools:
                    tool_def = _extract_tool_definition(tool)
                    if tool_def:
                        tool_definitions.append(tool_def)
                if tool_definitions:
                    span.set_attribute(
                        SpanAttributes.GEN_AI_TOOL_DEFINITIONS,
                        json.dumps(tool_definitions)
                    )

            result = wrapped(*args, **kwargs)

            # Set agent ID as name + hashed instance ID
            span.set_attribute(
                SpanAttributes.GEN_AI_AGENT_ID, _generate_agent_id(agent_name, result)
            )

            return result

    return wrapper
