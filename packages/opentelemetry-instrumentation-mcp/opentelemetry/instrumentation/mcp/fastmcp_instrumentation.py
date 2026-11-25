"""FastMCP-specific instrumentation logic."""

import json
import logging
import os

from opentelemetry.trace import Tracer
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues
from wrapt import register_post_import_hook, wrap_function_wrapper

from .utils import (
    dont_throw,
    guard_trace,
    guarded_span,
    record_error,
    record_tool_error,
    should_send_prompts,
)

logger = logging.getLogger(__name__)


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

        # Instrument FastMCP __init__ to capture server name. Hooked on the
        # defining module: resolving the lazy `fastmcp.FastMCP` attribute would
        # import server support, which a client-only install lacks.
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "fastmcp.server.server", "FastMCP.__init__", self._fastmcp_init_wrapper()
            ),
            "fastmcp.server.server",
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
        def capture_server_name(args, kwargs):
            if args and len(args) > 0:
                self._server_name = f"{args[0]}.mcp"
            elif 'name' in kwargs:
                self._server_name = f"{kwargs['name']}.mcp"

        def traced_method(wrapped, instance, args, kwargs):
            # Call the original __init__ first, outside the guard, so its
            # exceptions reach the caller.
            result = wrapped(*args, **kwargs)
            capture_server_name(args, kwargs)
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
        """Run one tool execution inside an mcp.server span and a nested tool span.

        Transparent to the traced tool: it gets the original args/kwargs, and the
        caller gets its exact result or exception. Every span operation is this
        instrumentation's own work, so its failures are logged, never raised.
        """
        if not self._tracer:
            return await wrapped(*args, **kwargs)

        entity_name = tool_key if tool_key else "unknown_tool"
        server_name = self._current_server_name()

        # Failures are recorded by hand below, with the message gated on
        # content capture; OTel's own exception recording would re-add that
        # text ungated on the way out.
        span_kwargs = {"record_exception": False, "set_status_on_exception": False}
        with guarded_span(
            logger, "mcp.server", self._tracer, "mcp.server", **span_kwargs
        ) as mcp_span, guarded_span(
            logger, "mcp.server.tool", self._tracer, f"{entity_name}.tool", **span_kwargs
        ) as tool_span:
            with guard_trace(logger, "mcp.server.tool attributes"):
                mcp_span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, "server")
                mcp_span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, "mcp.server")
                tool_span.set_attribute(
                    SpanAttributes.TRACELOOP_SPAN_KIND, TraceloopSpanKindValues.TOOL.value
                )
                tool_span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
                if server_name:
                    mcp_span.set_attribute(SpanAttributes.TRACELOOP_WORKFLOW_NAME, server_name)
                    tool_span.set_attribute(SpanAttributes.TRACELOOP_WORKFLOW_NAME, server_name)
                if self._should_send_prompts():
                    input_data = {"tool_name": entity_name, "arguments": tool_arguments or {}}
                    json_input = self._to_json(input_data)
                    if json_input is not None:
                        tool_span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json_input)

            try:
                result = await wrapped(*args, **kwargs)
            except Exception as e:
                with guard_trace(logger, "mcp.server.tool record"):
                    record_error(tool_span, e)
                    record_error(mcp_span, e)
                raise

            # Status first, in its own guard: a failure recording the output
            # must not cost the spans the attribute that says how the call went.
            with guard_trace(logger, "mcp.server.tool status"):
                # A tool can fail without raising: fastmcp 3 returns
                # ToolResult(is_error=True), MCP a CallToolResult(isError=True).
                if getattr(result, "is_error", False) or getattr(result, "isError", False):
                    content = getattr(result, "content", None)
                    record_tool_error(tool_span, content)
                    record_tool_error(mcp_span, content)
                else:
                    tool_span.set_status(Status(StatusCode.OK))
                    mcp_span.set_status(Status(StatusCode.OK))

            with guard_trace(logger, "mcp.server.tool output"):
                if self._should_send_prompts() and result:
                    # result.content for fastmcp 2.12.2+ and 3; older fastmcp
                    # returned the content list itself.
                    items = getattr(result, "content", result)
                    if isinstance(items, (list, tuple)):
                        output_data = []
                        for item in items:
                            if hasattr(item, "text"):
                                output_data.append({"type": "text", "content": item.text})
                            elif hasattr(item, "__dict__"):
                                output_data.append(item.__dict__)
                            else:
                                output_data.append(str(item))
                        output = self._to_json(output_data)
                        if output is not None:
                            tool_span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, output)
                            mcp_span.set_attribute(SpanAttributes.MCP_RESPONSE_VALUE, output)

            return result

    def _should_send_prompts(self):
        """Check if content tracing is enabled (matches traceloop SDK)"""
        return should_send_prompts()

    def _to_json(self, data):
        """Truncated JSON for a span attribute, or None if `data` doesn't serialize.

        Content that isn't JSON-serializable is simply not recorded, quietly:
        it is a property of the tool's data, not a tracing failure.
        """
        try:
            return self._truncate_json_if_needed(json.dumps(data))
        except (TypeError, ValueError):
            return None

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
