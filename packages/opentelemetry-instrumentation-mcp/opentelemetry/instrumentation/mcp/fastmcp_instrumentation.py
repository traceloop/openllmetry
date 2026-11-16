"""FastMCP-specific instrumentation logic."""

import json
import os

from opentelemetry.trace import Tracer
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues
from wrapt import register_post_import_hook, wrap_function_wrapper

from .utils import dont_throw, record_error, record_tool_error, should_send_prompts


class FastMCPInstrumentor:
    """Handles FastMCP-specific instrumentation logic."""

    def __init__(self):
        self._tracer = None
        self._server_name = None

    def instrument(self, tracer: Tracer):
        """Apply FastMCP-specific instrumentation."""
        self._tracer = tracer

        # Instrument FastMCP server-side tool execution (fastmcp 2)
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "fastmcp.tools.tool_manager", "ToolManager.call_tool", self._fastmcp_tool_wrapper()
            ),
            "fastmcp.tools.tool_manager",
        )

        # fastmcp 3 removed ToolManager; every tool runs through Tool._run.
        register_post_import_hook(self._wrap_fastmcp3_tool_run, "fastmcp.tools.base")

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

    def _wrap_fastmcp3_tool_run(self, module):
        tool_cls = getattr(module, "Tool", None)
        if tool_cls is None or not hasattr(tool_cls, "_run"):
            return

        # Tool._run(arguments, task_meta=None) is where a tool actually runs,
        # whichever route reached it (middleware, mounted servers, transforms).
        # A mounted server's FastMCPProviderTool overrides _run to delegate to
        # the child server, so only the child's real run is traced: one span
        # pair per execution.
        async def traced_run(wrapped, instance, args, kwargs):
            task_meta = kwargs.get("task_meta", args[1] if len(args) > 1 else None)
            if task_meta is not None:
                # Background task: _run only submits it and returns a
                # CreateTaskResult; the tool itself runs later in a worker.
                return await wrapped(*args, **kwargs)
            arguments = kwargs.get("arguments", args[0] if args else None)
            return await self._trace_tool_call(
                wrapped, args, kwargs, getattr(instance, "name", None), arguments
            )

        wrap_function_wrapper(module, "Tool._run", traced_run)

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
        """Create wrapper for fastmcp 2's ToolManager.call_tool(key, arguments)."""
        async def traced_method(wrapped, instance, args, kwargs):
            tool_key = kwargs.get("key", args[0] if args else None)
            tool_arguments = kwargs.get("arguments", args[1] if len(args) > 1 else None)
            return await self._trace_tool_call(wrapped, args, kwargs, tool_key, tool_arguments)

        return traced_method

    def _current_server_name(self):
        """The `<name>.mcp` of the server running the current tool call.

        Read from fastmcp's request context, so a mounted child server's tools
        are attributed to the child. Falls back to the last-constructed server.
        """
        try:
            from fastmcp.server.dependencies import get_context

            return f"{get_context().fastmcp.name}.mcp"
        except Exception:
            return self._server_name

    async def _trace_tool_call(self, wrapped, args, kwargs, tool_key, tool_arguments):
        """Run one tool execution inside an mcp.server span and a nested tool span."""
        if not self._tracer:
            return await wrapped(*args, **kwargs)

        tool_arguments = tool_arguments or {}
        entity_name = tool_key if tool_key else "unknown_tool"
        server_name = self._current_server_name()

        # Create parent server.mcp span
        # The error paths below record the failure themselves, with the
        # message gated on content capture. OTel's own exception recording
        # would re-add that text ungated on the way out.
        with self._tracer.start_as_current_span(
            "mcp.server", record_exception=False, set_status_on_exception=False
        ) as mcp_span:
            mcp_span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, "server")
            mcp_span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, "mcp.server")
            if server_name:
                mcp_span.set_attribute(SpanAttributes.TRACELOOP_WORKFLOW_NAME, server_name)

            # Create nested tool span
            span_name = f"{entity_name}.tool"
            with self._tracer.start_as_current_span(
                span_name, record_exception=False, set_status_on_exception=False
            ) as tool_span:
                tool_span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, TraceloopSpanKindValues.TOOL.value)
                tool_span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
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
                except Exception as e:
                    record_error(tool_span, e)
                    record_error(mcp_span, e)
                    raise

                try:
                    # Add output in traceloop format to tool span
                    if self._should_send_prompts() and result:
                        try:
                            # Convert FastMCP Content objects to serializable format
                            # Note: result.content for fastmcp 2.12.2+, fallback to result for older versions
                            output_data = []
                            result_items = getattr(result, "content", result)
                            if not isinstance(result_items, (list, tuple)):
                                result_items = []
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

                    # A tool can fail without raising: fastmcp 3 returns
                    # ToolResult(is_error=True), MCP a CallToolResult(isError=True).
                    if getattr(result, "is_error", False) or getattr(result, "isError", False):
                        content = getattr(result, "content", None)
                        record_tool_error(tool_span, content)
                        record_tool_error(mcp_span, content)
                    else:
                        tool_span.set_status(Status(StatusCode.OK))
                        mcp_span.set_status(Status(StatusCode.OK))
                except Exception:
                    pass
                return result

    def _should_send_prompts(self):
        """Check if content tracing is enabled (matches traceloop SDK)"""
        return should_send_prompts()

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
