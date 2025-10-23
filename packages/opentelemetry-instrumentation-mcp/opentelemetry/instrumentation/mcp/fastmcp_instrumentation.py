"""FastMCP-specific instrumentation logic."""

import json
import os

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
        self._server_name = None

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

        # Instrument FastMCP __init__ to capture server name
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "fastmcp", "FastMCP.__init__", self._fastmcp_init_wrapper()
            ),
            "fastmcp",
        )

    def uninstrument(self):
        """Remove FastMCP-specific instrumentation."""
        # Note: wrapt doesn't provide a clean way to unwrap post-import hooks
        # This is a limitation we'll need to document
        pass

    def _fastmcp_init_wrapper(self):
        """Create wrapper for FastMCP initialization to capture server name."""
        @dont_throw
        def traced_method(wrapped, instance, args, kwargs):
            # Call the original __init__ first
            result = wrapped(*args, **kwargs)

            if args and len(args) > 0:
                self._server_name = f"{args[0]}.mcp"
            elif 'name' in kwargs:
                self._server_name = f"{kwargs['name']}.mcp"

            return result
        return traced_method

    def _fastmcp_tool_wrapper(self):
        """Create wrapper for FastMCP tool execution."""
        async def traced_method(wrapped, instance, args, kwargs):
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

            # Create parent server.mcp span
            with self._tracer.start_as_current_span("mcp.server") as mcp_span:
                mcp_span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, "server")
                mcp_span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, "mcp.server")
                if self._server_name:
                    mcp_span.set_attribute(SpanAttributes.TRACELOOP_WORKFLOW_NAME, self._server_name)

                # Create nested tool span
                span_name = f"{entity_name}.tool"
                with self._tracer.start_as_current_span(span_name) as tool_span:
                    tool_span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, TraceloopSpanKindValues.TOOL.value)
                    tool_span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
                    if self._server_name:
                        tool_span.set_attribute(SpanAttributes.TRACELOOP_WORKFLOW_NAME, self._server_name)

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
                    except Exception as e:
                        tool_span.set_attribute(ERROR_TYPE, type(e).__name__)
                        tool_span.record_exception(e)
                        tool_span.set_status(Status(StatusCode.ERROR, str(e)))

                        mcp_span.set_attribute(ERROR_TYPE, type(e).__name__)
                        mcp_span.record_exception(e)
                        mcp_span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

                    try:
                        # Add output in traceloop format to tool span
                        if self._should_send_prompts() and result:
                            try:
                                # Convert FastMCP Content objects to serializable format
                                # Note: result.content for fastmcp 2.12.2+, fallback to result for older versions
                                output_data = []
                                result_items = result.content if hasattr(result, 'content') else result
                                for item in result_items:
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
                    except Exception:
                        pass
                    return result

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
