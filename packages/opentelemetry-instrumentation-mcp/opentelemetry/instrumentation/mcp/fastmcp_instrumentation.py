"""FastMCP-specific instrumentation logic."""

import json
import os
import gc
import inspect

from opentelemetry.trace import Tracer
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from wrapt import register_post_import_hook, wrap_function_wrapper

from .utils import dont_throw


class FastMCPInstrumentor:
    """Handles FastMCP-specific instrumentation logic."""

    def __init__(self):
        self._tracer = None

    def _find_fastmcp_server_name(self, instance):
        """Find the FastMCP server name by inspecting the call stack and garbage collector."""
        try:
            # First, try to find the server by checking if this tool manager belongs to a FastMCP server
            for obj in gc.get_objects():
                # Look for FastMCP instances that have this tool manager
                print("NOMI - _find_fastmcp_server_name: obj:", obj)
                if (hasattr(obj, '__class__') and
                        obj.__class__.__name__ == 'FastMCP' and
                        hasattr(obj, '_tool_manager') and
                        obj._tool_manager is instance):
                    if hasattr(obj, 'name') and obj.name:
                        print("NOMI - _find_fastmcp_server_name: obj.name:", obj.name)
                        return obj.name
                    break

            # Fallback: Try to find any FastMCP server in the current frame stack
            current_frame = inspect.currentframe()
            while current_frame:
                for local_var in current_frame.f_locals.values():
                    if (hasattr(local_var, '__class__') and
                            local_var.__class__.__name__ == 'FastMCP' and
                            hasattr(local_var, 'name') and local_var.name):
                        print("NOMI - _find_fastmcp_server_name: local_var.name:", local_var.name)
                        return local_var.name
                current_frame = current_frame.f_back

        except Exception:
            pass

        return None

    def instrument(self, tracer: Tracer):
        """Apply FastMCP-specific instrumentation."""
        self._tracer = tracer

        # Instrument FastMCP server-side tool execution
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "fastmcp.tools.tool_manager", "ToolManager.call_tool", self._fastmcp_tool_wrapper()
            ),
            "fastmcp.tools.tool_manager",
        )

    def uninstrument(self):
        """Remove FastMCP-specific instrumentation."""
        # Note: wrapt doesn't provide a clean way to unwrap post-import hooks
        # This is a limitation we'll need to document
        pass

    def _fastmcp_tool_wrapper(self):
        """Create wrapper for FastMCP tool execution."""
        @dont_throw
        async def traced_method(wrapped, instance, args, kwargs):
            print("NOMI - _fastmcp_tool_wrapper: args:", args)
            print("NOMI - _fastmcp_tool_wrapper: kwargs:", kwargs)
            if not self._tracer:
                return await wrapped(*args, **kwargs)

            # Extract tool name from arguments - FastMCP has different call patterns
            tool_key = None
            tool_arguments = {}

            # Pattern 1: kwargs with 'key' parameter
            if kwargs and 'key' in kwargs:
                tool_key = kwargs.get('key')
                tool_arguments = kwargs.get('arguments', {})
            # Pattern 2: positional args (tool_name, arguments)
            elif args and len(args) >= 1:
                tool_key = args[0]
                tool_arguments = args[1] if len(args) > 1 else {}

            entity_name = tool_key if tool_key else "unknown_tool"

            # Best practice: read FastMCP server name from execution context
            context = None
            if kwargs and 'context' in kwargs:
                context = kwargs.get('context')
            elif args and len(args) >= 3:
                # call_tool(self, key, arguments, context=...)
                context = args[2]

            server_name = None
            if context is not None:
                fastmcp_obj = getattr(context, 'fastmcp', None)
                if fastmcp_obj is not None:
                    server_name = getattr(fastmcp_obj, 'name', None)
            print("NOMI - _fastmcp_tool_wrapper: server_name:", server_name)

            # Create parent server.mcp span
            with self._tracer.start_as_current_span("mcp.server") as mcp_span:
                mcp_span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, "server")
                mcp_span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, "mcp.server")

                # # Try to set the FastMCP server name as workflow name (prefer context, fallback to discovery)
                # if not server_name:
                #     server_name = self._find_fastmcp_server_name(instance)
                if server_name:
                    mcp_span.set_attribute(SpanAttributes.TRACELOOP_WORKFLOW_NAME, f"{server_name}.mcp")

                # Create nested tool span
                span_name = f"{entity_name}.tool"
                with self._tracer.start_as_current_span(span_name) as tool_span:
                    tool_span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, TraceloopSpanKindValues.TOOL.value)
                    tool_span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)

                    # Set workflow name on tool span as well if we found it
                    if server_name:
                        tool_span.set_attribute(SpanAttributes.TRACELOOP_WORKFLOW_NAME, server_name)

                    if self._should_send_prompts():
                        try:
                            input_data = {
                                "tool_name": entity_name,
                                "arguments": tool_arguments
                            }
                            json_input = json.dumps(input_data, cls=self._get_json_encoder())
                            truncated_input = self._truncate_json_if_needed(json_input)
                            tool_span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, truncated_input)
                        except (TypeError, ValueError):
                            pass  # Skip input logging if serialization fails

                    try:
                        result = await wrapped(*args, **kwargs)

                        # Add output in traceloop format to tool span
                        if self._should_send_prompts() and result:
                            try:
                                # Convert FastMCP Content objects to serializable format
                                output_data = []
                                for item in result:
                                    if hasattr(item, 'text'):
                                        output_data.append({"type": "text", "content": item.text})
                                    elif hasattr(item, '__dict__'):
                                        output_data.append(item.__dict__)
                                    else:
                                        output_data.append(str(item))

                                json_output = json.dumps(output_data, cls=self._get_json_encoder())
                                truncated_output = self._truncate_json_if_needed(json_output)
                                tool_span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, truncated_output)

                                # Also add response to MCP span
                                mcp_span.set_attribute(SpanAttributes.MCP_RESPONSE_VALUE, truncated_output)
                            except (TypeError, ValueError):
                                pass  # Skip output logging if serialization fails

                        tool_span.set_status(Status(StatusCode.OK))
                        mcp_span.set_status(Status(StatusCode.OK))
                        return result

                    except Exception as e:
                        tool_span.set_attribute(ERROR_TYPE, type(e).__name__)
                        tool_span.record_exception(e)
                        tool_span.set_status(Status(StatusCode.ERROR, str(e)))

                        mcp_span.set_attribute(ERROR_TYPE, type(e).__name__)
                        mcp_span.record_exception(e)
                        mcp_span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

        return traced_method

    def _should_send_prompts(self):
        """Check if content tracing is enabled (matches traceloop SDK)"""
        return (
            os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
        ).lower() == "true"

    def _get_json_encoder(self):
        """Get JSON encoder class (simplified - traceloop SDK uses custom JSONEncoder)"""
        return None  # Use default JSON encoder

    def _truncate_json_if_needed(self, json_str: str) -> str:
        """Truncate JSON if it exceeds OTEL limits (matches traceloop SDK)"""
        limit_str = os.getenv("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT")
        if limit_str:
            try:
                limit = int(limit_str)
                if limit > 0 and len(json_str) > limit:
                    return json_str[:limit]
            except ValueError:
                pass
        return json_str
