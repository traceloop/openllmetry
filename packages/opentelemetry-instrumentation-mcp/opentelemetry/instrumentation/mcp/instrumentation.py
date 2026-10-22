import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional, Callable, Collection, Tuple, Union, cast
import json
import logging
import weakref

from opentelemetry import context, propagate
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import get_tracer, Tracer
from wrapt import ObjectProxy
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE

from opentelemetry.instrumentation.mcp.version import __version__
from opentelemetry.instrumentation.mcp._wrapping import _InstrumentedMethods
from opentelemetry.instrumentation.mcp.utils import (
    Config,
    dont_throw,
    error_status,
    guard_trace,
    record_error,
    should_send_prompts,
)
from opentelemetry.instrumentation.mcp.fastmcp_instrumentation import (
    FastMCPInstrumentor,
)

_instruments = ("mcp >= 1.6.0",)

logger = logging.getLogger(__name__)


@dataclass
class _SessionEntry:
    """One task's open `mcp.client.session` span on a FastMCP client.

    FastMCP's Client is reentrant: nested `async with client` reuses one
    session, so only the outermost entry owns the span and `depth` counts
    the entries still open.
    """

    context_manager: Any
    span: Any
    depth: int = 1


def _task_key() -> Optional[asyncio.Task]:
    """The running task, which owns the OTel context token the span attached.

    A context token can only be reset in the contextvars Context that created
    it, so a session span must be ended by the task that started it. None when
    there is no asyncio task (e.g. a different async backend): one shared slot.
    """
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def _finished_task_entry(sessions):
    """The entry of a task that has already finished, or (None, None).

    `async with` enters and exits in one task, but a manual __aenter__ and
    __aexit__ need not: pytest-asyncio runs a fixture's setup and teardown in
    separate tasks. The enter's task is then done by the time of the exit.
    """
    for key, entry in sessions.items():
        if key is not None and key.done():
            return key, entry
    return None, None


# Per client, but kept off the client itself: Client.new() is a copy.copy,
# which would share an attribute's dict between the original and the copy.
_client_sessions = weakref.WeakKeyDictionary()


class McpInstrumentor(BaseInstrumentor):
    def __init__(self, exception_logger=None):
        super().__init__()
        Config.exception_logger = exception_logger
        # BaseInstrumentor is a singleton, but Python calls __init__ on every
        # construction. Keep the helper that owns the installed FastMCP patches.
        if not hasattr(self, "_fastmcp_instrumentor"):
            self._fastmcp_instrumentor = FastMCPInstrumentor()
            self._methods = _InstrumentedMethods()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        # Instrument FastMCP
        self._fastmcp_instrumentor.instrument(tracer)

        # Instrument FastMCP Client to create a session-level span
        self._methods.wrap(
            "fastmcp.client",
            "Client.__aenter__",
            self._fastmcp_client_enter_wrapper(tracer),
        )
        self._methods.wrap(
            "fastmcp.client",
            "Client.__aexit__",
            self._fastmcp_client_exit_wrapper(tracer),
        )

        self._methods.wrap(
            "mcp.client.sse", "sse_client", self._transport_wrapper(tracer)
        )
        self._methods.wrap(
            "mcp.server.sse",
            "SseServerTransport.connect_sse",
            self._transport_wrapper(tracer),
        )
        self._methods.wrap(
            "mcp.client.stdio", "stdio_client", self._transport_wrapper(tracer)
        )
        self._methods.wrap(
            "mcp.server.stdio", "stdio_server", self._transport_wrapper(tracer)
        )
        self._methods.wrap(
            "mcp.server.session",
            "ServerSession.__init__",
            self._base_session_init_wrapper(tracer),
        )
        self._methods.wrap(
            "mcp.client.streamable_http",
            "streamablehttp_client",
            self._transport_wrapper(tracer),
        )
        self._methods.wrap(
            "mcp.server.streamable_http",
            "StreamableHTTPServerTransport.connect",
            self._transport_wrapper(tracer),
        )
        self._methods.wrap(
            "mcp.shared.session",
            "BaseSession.send_request",
            self.patch_mcp_client(tracer),
        )

    def _uninstrument(self, **kwargs):
        self._methods.unwrap()
        self._fastmcp_instrumentor.uninstrument()

    def _transport_wrapper(self, tracer):
        @asynccontextmanager
        async def traced_method(
            wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any
        ) -> AsyncGenerator[
            Union[
                Tuple[InstrumentedStreamReader, InstrumentedStreamWriter],
                Tuple[InstrumentedStreamReader, InstrumentedStreamWriter, Any],
            ],
            None,
        ]:
            async with wrapped(*args, **kwargs) as result:
                try:
                    read_stream, write_stream = result
                    yield InstrumentedStreamReader(
                        read_stream, tracer
                    ), InstrumentedStreamWriter(write_stream, tracer)
                except ValueError:
                    try:
                        read_stream, write_stream, get_session_id_callback = result
                        yield InstrumentedStreamReader(
                            read_stream, tracer
                        ), InstrumentedStreamWriter(
                            write_stream, tracer
                        ), get_session_id_callback
                    except Exception as e:
                        logging.warning(
                            f"mcp instrumentation _transport_wrapper exception: {e}"
                        )
                        yield result
                except Exception as e:
                    logging.warning(
                        f"mcp instrumentation transport_wrapper exception: {e}"
                    )
                    yield result

        return traced_method

    def _base_session_init_wrapper(self, tracer):
        def traced_method(
            wrapped: Callable[..., None], instance: Any, args: Any, kwargs: Any
        ) -> None:
            wrapped(*args, **kwargs)
            reader = getattr(instance, "_incoming_message_stream_reader", None)
            writer = getattr(instance, "_incoming_message_stream_writer", None)
            if reader and writer:
                setattr(
                    instance,
                    "_incoming_message_stream_reader",
                    ContextAttachingStreamReader(reader, tracer),
                )
                setattr(
                    instance,
                    "_incoming_message_stream_writer",
                    ContextSavingStreamWriter(writer, tracer),
                )

        return traced_method

    def patch_mcp_client(self, tracer: Tracer):
        @dont_throw
        async def traced_method(wrapped, instance, args, kwargs):
            meta = None
            method = None
            params = None
            if len(args) > 0 and hasattr(args[0].root, "method"):
                method = args[0].root.method
            if len(args) > 0 and hasattr(args[0].root, "params"):
                params = args[0].root.params
            if params:
                if hasattr(args[0].root.params, "meta"):
                    meta = args[0].root.params.meta

            # Handle trace context propagation
            if meta and len(args) > 0:
                carrier = {}
                TraceContextTextMapPropagator().inject(carrier)
                meta.traceparent = carrier["traceparent"]
                args[0].root.params.meta = meta

            # Create different span types based on method
            if method == "tools/call":
                return await self._handle_tool_call(
                    tracer, method, params, args, kwargs, wrapped
                )
            else:
                return await self._handle_mcp_method(
                    tracer, method, args, kwargs, wrapped
                )

        return traced_method

    def _fastmcp_client_enter_wrapper(self, tracer):
        """Wrapper for FastMCP Client.__aenter__ to start a session trace"""

        async def traced_method(wrapped, instance, args, kwargs):
            key = _task_key()
            sessions = _client_sessions.get(instance)
            entry = sessions.get(key) if sessions is not None else None
            if entry is not None:
                # Nested entry: FastMCP reuses its session, so reuse the span.
                entry.depth += 1
                try:
                    return await wrapped(*args, **kwargs)
                except BaseException:
                    # A failed __aenter__ gets no __aexit__ from `async with`,
                    # so it must not count as an open entry.
                    entry.depth -= 1
                    raise

            # Open a root span for the MCP client session and make it current.
            # Guarded, because all of it is this instrumentation's own work: if
            # the span cannot be opened, `span is None` below means "untraced",
            # never "do not connect".
            span_context_manager = None
            span = None
            with guard_trace(logger, "mcp.client.session start"):
                span_context_manager = tracer.start_as_current_span(
                    "mcp.client.session",
                    record_exception=False,
                    set_status_on_exception=False,
                )
                span = span_context_manager.__enter__()
                span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, "session")
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_NAME, "mcp.client.session"
                )

            # `span` is the gate throughout: it is set only once __enter__ has
            # succeeded, so it also stands for "there is something to end".
            if span is not None:
                # Kept per task, not per instance: the exit has to detach this
                # span's context token from the task that attached it. The span
                # is kept so the exit can report a teardown failure on it.
                if sessions is None:
                    sessions = _client_sessions.setdefault(instance, {})
                sessions[key] = _SessionEntry(span_context_manager, span)

            entered = False
            try:
                # Call the original method. Nothing here catches what it raises:
                # swallowing it would run the `async with` body with `c = None`.
                result = await wrapped(*args, **kwargs)
                entered = True
                return result
            except Exception as e:
                if span is not None:
                    with guard_trace(logger, "mcp.client.session record"):
                        record_error(span, e)
                raise
            finally:
                # A failed __aenter__ means `async with` never calls __aexit__,
                # so end the span here, detaching it from the current context.
                # A finally rather than the except, so a cancellation (a
                # BaseException, not an error) ends it too, UNSET. The entry is
                # dropped first, so a span end failure can't leave it behind.
                if not entered and span is not None:
                    sessions.pop(key, None)
                    with guard_trace(logger, "mcp.client.session end"):
                        span_context_manager.__exit__(None, None, None)

        return traced_method

    def _fastmcp_client_exit_wrapper(self, tracer):
        """Wrapper for FastMCP Client.__aexit__ to end the session trace"""

        async def traced_method(wrapped, instance, args, kwargs):
            sessions = _client_sessions.get(instance) or {}
            key = _task_key()
            entry = sessions.get(key)
            if entry is None:
                # Entered in a task that has since finished, or not at all (the
                # span failed to open): then there is no span of ours to end.
                key, entry = _finished_task_entry(sessions)
            try:
                # Call the original method first. Its failure reaches the caller:
                # swallowing it would report a clean teardown for a dead transport.
                return await wrapped(*args, **kwargs)
            except Exception as e:
                # Record the teardown failure before __exit__ ends the span --
                # the span's own exception recording is off, so nothing else
                # would report it. Guarded, so recording cannot take the place
                # of the failure being re-raised.
                if entry is not None:
                    with guard_trace(logger, "mcp.client.session record"):
                        record_error(entry.span, e)
                raise
            finally:
                # End the span in a finally so a cancelled __aexit__ (a
                # BaseException, not an error) still ends it, UNSET. Only the
                # outermost exit ends it; nested exits just unwind the depth.
                # The entry is dropped before the guarded end, so a span end
                # failure can't leave it behind for the next `async with`.
                if entry is not None:
                    entry.depth -= 1
                    if entry.depth <= 0:
                        sessions.pop(key, None)
                        # Always close the context manager, even from another
                        # task: ending the span alone would leave its generator
                        # to be closed by GC, ending the span a second time.
                        with guard_trace(logger, "mcp.client.session end"):
                            entry.context_manager.__exit__(None, None, None)

        return traced_method

    async def _handle_tool_call(self, tracer, method, params, args, kwargs, wrapped):
        """Handle tools/call with tool semantics"""
        # Extract the actual tool name
        entity_name = method
        span_name = f"{method}.tool"
        if params:
            try:
                if hasattr(params, "name"):
                    entity_name = params.name
                    span_name = f"{params.name}.tool"
                elif hasattr(params, "__dict__") and "name" in params.__dict__:
                    entity_name = params.__dict__["name"]
                    span_name = f"{params.__dict__['name']}.tool"
            except Exception:
                pass

        # _execute_and_handle_result records the failure itself, with the
        # message gated on content capture; OTel's own exception recording
        # would re-add that text ungated on the way out.
        with tracer.start_as_current_span(
            span_name, record_exception=False, set_status_on_exception=False
        ) as span:
            # Set tool-specific attributes
            span.set_attribute(
                SpanAttributes.TRACELOOP_SPAN_KIND, TraceloopSpanKindValues.TOOL.value
            )
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)

            # Add input. Tool arguments are request content, so they are
            # recorded only when content capture is enabled.
            clean_input = (
                self._extract_clean_input(method, params)
                if should_send_prompts()
                else None
            )
            if clean_input:
                try:
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_INPUT, json.dumps(clean_input)
                    )
                except (TypeError, ValueError):
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_INPUT, str(clean_input)
                    )

            return await self._execute_and_handle_result(
                span, method, args, kwargs, wrapped, clean_output=True
            )

    async def _handle_mcp_method(self, tracer, method, args, kwargs, wrapped):
        """Handle non-tool MCP methods with simple serialization"""
        with tracer.start_as_current_span(
            f"{method}.mcp", record_exception=False, set_status_on_exception=False
        ) as span:
            # The serialized request is content: it carries caller-supplied
            # params, so it is recorded only when content capture is enabled.
            if should_send_prompts():
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_INPUT, f"{serialize(args[0])}"
                )
            return await self._execute_and_handle_result(
                span, method, args, kwargs, wrapped, clean_output=False
            )

    async def _execute_and_handle_result(
        self, span, method, args, kwargs, wrapped, clean_output=False
    ):
        """Execute the wrapped function and handle the result"""
        try:
            result = await wrapped(*args, **kwargs)
            # Add output. The response body is content, so it is recorded only
            # when content capture is enabled.
            if not should_send_prompts():
                pass
            elif clean_output:
                clean_output_data = self._extract_clean_output(method, result)
                if clean_output_data:
                    try:
                        span.set_attribute(
                            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                            json.dumps(clean_output_data),
                        )
                    except (TypeError, ValueError):
                        span.set_attribute(
                            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                            str(clean_output_data),
                        )
            else:
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT, serialize(result)
                )
            # Handle errors
            if hasattr(result, "isError") and result.isError:
                span.set_attribute(ERROR_TYPE, "tool_error")
                if len(result.content) > 0:
                    span.set_status(error_status(f"{result.content[0].text}"))
            else:
                span.set_status(Status(StatusCode.OK))
            return result
        except Exception as e:
            record_error(span, e)
            raise

    def _extract_clean_input(self, method: str, params: Any) -> dict:
        """Extract clean input parameters for different MCP method types"""
        if not params:
            return {}

        try:
            if method == "tools/call":
                # For tool calls, extract name and arguments
                result = {}
                if hasattr(params, "name"):
                    result["tool_name"] = params.name
                if hasattr(params, "arguments"):
                    if hasattr(params.arguments, "__dict__"):
                        result["arguments"] = params.arguments.__dict__
                    else:
                        result["arguments"] = params.arguments
                elif hasattr(params, "__dict__") and "arguments" in params.__dict__:
                    result["arguments"] = params.__dict__["arguments"]
                return result
            elif method == "tools/list":
                # For list_tools, there are usually no parameters
                return {}
            else:
                # For other methods, try to serialize params cleanly
                if hasattr(params, "__dict__"):
                    # Remove internal fields starting with _ and non-serializable objects
                    clean_params = {}
                    for k, v in params.__dict__.items():
                        if not k.startswith("_"):
                            try:
                                # Test if the value is JSON serializable
                                json.dumps(v)
                                clean_params[k] = v
                            except (TypeError, ValueError):
                                # If not serializable, store a string representation
                                clean_params[k] = str(type(v).__name__)
                    return clean_params
                else:
                    return {"params": str(params)}
        except Exception:
            return {}

    def _extract_clean_output(self, method: str, result: Any) -> dict:
        """Extract clean output for different MCP method types"""
        if not result:
            return {}

        try:
            if method == "tools/call":
                # For tool calls, extract the actual result content
                output = {}
                if hasattr(result, "content") and result.content:
                    if len(result.content) > 0:
                        content_item = result.content[0]
                        if hasattr(content_item, "text"):
                            output["result"] = content_item.text
                        elif hasattr(content_item, "__dict__"):
                            output["result"] = content_item.__dict__
                        else:
                            output["result"] = str(content_item)

                # Check if this is an error response
                if hasattr(result, "isError") and result.isError:
                    output["is_error"] = True

                return output
            elif method == "tools/list":
                # For list_tools, extract tool names and descriptions
                output = {"tools": []}
                if hasattr(result, "tools") and result.tools:
                    for tool in result.tools:
                        tool_info = {}
                        if hasattr(tool, "name"):
                            tool_info["name"] = tool.name
                        if hasattr(tool, "description"):
                            tool_info["description"] = tool.description
                        output["tools"].append(tool_info)
                return output
            else:
                # For other methods, try to serialize result cleanly
                if hasattr(result, "__dict__"):
                    clean_result = {
                        k: v
                        for k, v in result.__dict__.items()
                        if not k.startswith("_")
                    }
                    return clean_result
                else:
                    return {"result": str(result)}
        except Exception:
            return {}


def serialize(request, depth=0, max_depth=4):
    """Serialize input args to MCP server into JSON.
    The function accepts input object and converts into JSON
    keeping depth in mind to prevent creating large nested JSON"""
    if depth > max_depth:
        return {}
    depth += 1

    def is_serializable(request):
        try:
            json.dumps(request)
            return True
        except Exception:
            return False

    if is_serializable(request):
        return json.dumps(request)
    else:
        result = {}
        try:
            if hasattr(request, "model_dump_json"):
                return request.model_dump_json()
            if hasattr(request, "__dict__"):
                for attrib in request.__dict__:
                    if not attrib.startswith("_"):
                        if type(request.__dict__[attrib]) in [
                            bool,
                            str,
                            int,
                            float,
                            type(None),
                        ]:
                            result[str(attrib)] = request.__dict__[attrib]
                        else:
                            result[str(attrib)] = serialize(
                                request.__dict__[attrib], depth
                            )
        except Exception:
            pass
        return json.dumps(result)


class InstrumentedStreamReader(ObjectProxy):  # type: ignore
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    def __init__(self, wrapped, tracer):
        super().__init__(wrapped)
        self._tracer = tracer

    async def __aenter__(self) -> Any:
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    @dont_throw
    async def __aiter__(self) -> AsyncGenerator[Any, None]:
        from mcp.types import JSONRPCMessage, JSONRPCRequest

        async for item in self.__wrapped__:
            # Handle different item types based on what's available
            request = None
            if hasattr(item, "message") and hasattr(item.message, "root"):
                request = item.message.root
            elif type(item) is JSONRPCMessage:
                request = cast(JSONRPCMessage, item).root
            elif hasattr(item, "root"):
                request = item.root
            else:
                yield item
                continue

            if not isinstance(request, JSONRPCRequest):
                yield item
                continue

            if request.params:
                meta = request.params.get("_meta")
                if meta:
                    ctx = propagate.extract(meta)
                    restore = context.attach(ctx)
                    try:
                        yield item
                        continue
                    finally:
                        context.detach(restore)
            yield item


class InstrumentedStreamWriter(ObjectProxy):  # type: ignore
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    def __init__(self, wrapped, tracer):
        super().__init__(wrapped)
        self._tracer = tracer

    async def __aenter__(self) -> Any:
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    async def send(self, item: Any) -> Any:
        from mcp.types import JSONRPCMessage, JSONRPCRequest

        # All span work happens up front, inside the guard: a failure costs the
        # span, never the message. `span is None` means this send is untraced.
        span_context_manager = None
        span = None
        with guard_trace(logger, "ResponseStreamWriter"):
            # Handle different item types based on what's available
            request = None
            if hasattr(item, "message") and hasattr(item.message, "root"):
                request = item.message.root
            elif type(item) is JSONRPCMessage:
                request = cast(JSONRPCMessage, item).root
            elif hasattr(item, "root"):
                request = item.root

            if request is not None:
                # The wrapped send runs inside this span, so a transport failure
                # would be recorded by OTel with its message and stacktrace
                # ungated. The except below reports it instead, message gated.
                span_context_manager = self._tracer.start_as_current_span(
                    "ResponseStreamWriter",
                    record_exception=False,
                    set_status_on_exception=False,
                )
                span = span_context_manager.__enter__()

                if isinstance(request, JSONRPCRequest):
                    # Injection first: the server downstream depends on it, so it
                    # must not be lost to a failure in the attributes below.
                    if not request.params:
                        request.params = {}
                    meta = request.params.setdefault("_meta", {})
                    propagate.get_global_textmap().inject(meta)

                if hasattr(request, "result"):
                    # The response body is content, and so is the error text
                    # below -- it is the same payload.
                    if should_send_prompts():
                        span.set_attribute(
                            SpanAttributes.MCP_RESPONSE_VALUE,
                            f"{serialize(request.result)}",
                        )
                    if "isError" in request.result:
                        if request.result["isError"] is True:
                            span.set_status(
                                error_status(f"{request.result['content'][0]['text']}")
                            )
                if hasattr(request, "id"):
                    span.set_attribute(SpanAttributes.MCP_REQUEST_ID, f"{request.id}")

        try:
            # The transport call: one site, never guarded. Swallowing a failure
            # here would tell the caller a message went out when it did not.
            return await self.__wrapped__.send(item)
        except Exception as e:
            if span is not None:
                with guard_trace(logger, "ResponseStreamWriter record"):
                    record_error(span, e)
            raise
        finally:
            if span is not None:
                with guard_trace(logger, "ResponseStreamWriter end"):
                    span_context_manager.__exit__(None, None, None)


@dataclass(slots=True, frozen=True)
class ItemWithContext:
    item: Any
    ctx: context.Context


class ContextSavingStreamWriter(ObjectProxy):  # type: ignore
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    def __init__(self, wrapped, tracer):
        super().__init__(wrapped)
        self._tracer = tracer

    async def __aenter__(self) -> Any:
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    async def send(self, item: Any) -> Any:
        # Not guarded: nothing here is tracing work that can fail, and a
        # transport failure from the wrapped send has to reach the caller.
        # Removed RequestStreamWriter span creation - we don't need low-level protocol spans
        ctx = context.get_current()
        return await self.__wrapped__.send(ItemWithContext(item, ctx))


class ContextAttachingStreamReader(ObjectProxy):  # type: ignore
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    def __init__(self, wrapped, tracer):
        super().__init__(wrapped)
        self._tracer = tracer

    async def __aenter__(self) -> Any:
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    async def __aiter__(self) -> AsyncGenerator[Any, None]:
        async for item in self.__wrapped__:
            item_with_context = cast(ItemWithContext, item)
            restore = context.attach(item_with_context.ctx)
            try:
                yield item_with_context.item
            finally:
                context.detach(restore)
