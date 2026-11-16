"""Shared utilities for MCP instrumentation."""

import asyncio
import logging
import os
import traceback
from contextlib import contextmanager
from typing import Optional

from opentelemetry import context as context_api
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import INVALID_SPAN, Status, StatusCode


class Config:
    exception_logger = None


def should_send_prompts() -> bool:
    """Whether request/response content may be recorded on spans.

    Mirrors the traceloop SDK's ``TRACELOOP_TRACE_CONTENT`` switch: content
    capture is on unless an operator explicitly turns it off. Shared by the
    FastMCP server wrapper and the MCP client path so a single environment
    variable governs both, which is what the package README documents.

    Like the other instrumentations, the SDK's per-association content allow
    list can also enable capture via ``override_enable_content_tracing``.
    """
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or bool(context_api.get_value("override_enable_content_tracing"))


def frames_without_content(exc: BaseException) -> str:
    """`exc`'s stack as file, line and function -- without the frames' source lines.

    The frames are not content and are what you debug from; the source line is,
    because a raise site like ``ToolError("...")`` reproduces its own message
    there, and that message is the thing being withheld.
    """
    return "\n".join(
        f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}'
        for frame in traceback.extract_tb(exc.__traceback__)
    )


def error_status(description: str) -> Status:
    """An ERROR status, carrying `description` only when content capture is on.

    The status code is not content, but its description is: on the client path
    it is the server's own error text. Callers record the exception type
    separately, so the failure stays visible either way.
    """
    if should_send_prompts():
        return Status(StatusCode.ERROR, description)
    return Status(StatusCode.ERROR)


def error_detail(content) -> Optional[str]:
    """The error text of an MCP result's first content block, or a marker for one
    that has none.

    Only ``TextContent`` carries ``text``; an image, audio or embedded resource
    block is just as valid in an error result and has no message to quote, so
    name the kind of block instead of reaching for text that isn't there.
    ``content`` holds model objects on the client path and raw JSON dicts on the
    writer path, and may be absent or empty on either -- an error result with no
    content at all yields ``None``, meaning "an error, with nothing to say
    about it".
    """
    if not content:
        return None
    block = content[0]

    def field(name):
        if isinstance(block, dict):
            return block.get(name)
        return getattr(block, name, None)

    text = field("text")
    if text is not None:
        return f"{text}"
    return f"non-text content: {field('type') or type(block).__name__}"


def record_error(span, exc) -> None:
    """Mark `span` failed, withholding only the parts of `exc` that are content.

    The exception type and the stack frames are not content and are recorded
    either way -- the stacktrace is the part you debug from. The message is,
    and it reaches a traceback twice over: as the last line of a formatted
    one, and inside the source line of the raise site's own frame. Hence the
    frames are rendered by hand, without either.

    Both branches go through ``record_exception`` so the event's non-content
    fields (``exception.type``, ``exception.escaped``) are written by the SDK
    and match whichever way the switch is set; only the content fields are
    overridden.
    """
    span.set_attribute(ERROR_TYPE, type(exc).__name__)
    if should_send_prompts():
        span.record_exception(exc)
    else:
        span.record_exception(
            exc,
            attributes={
                "exception.message": "",
                "exception.stacktrace": frames_without_content(exc),
            },
        )
    span.set_status(error_status(str(exc)))


def log_trace_failure(logger: logging.Logger, origin: str, exc: BaseException) -> None:
    """Record a failure in the instrumentation's own code, then return.

    Shared by ``dont_throw`` and ``guard_trace`` so an instrumentation failure
    is reported the same way wherever it is caught: a log line, and the
    exception logger an operator configured.

    Logged at WARNING, not DEBUG: a swallowed failure is invisible from the
    traced program's point of view, so this line is the only record an operator
    gets that tracing stopped working. At DEBUG, every deployment running at
    INFO or above had none.

    The WARNING line carries only the exception; the stack goes to DEBUG. A
    failure on a per-message path repeats for every message, and a traceback
    apiece turns a visible warning into a flood.

    The exception's message is withheld when content capture is off, on the same
    terms as ``record_error`` withholds it from a span: a guard wraps code that
    handles caller data, and an exception raised in there can quote it back --
    a pydantic ``ValidationError`` names the input value it rejected. The type
    and the frames are not content, so what the line is for -- an operator
    seeing that tracing broke, and where -- survives either way.

    The operator's ``exception_logger`` is instrumentation too, so its own
    failure is contained here rather than raised into the traced program --
    this function is called from inside every guard, and anything that escapes
    it escapes them all.
    """
    if should_send_prompts():
        logger.warning(
            "OpenLLMetry failed to trace in %s, error: %s: %s",
            origin,
            type(exc).__name__,
            exc,
        )
        logger.debug("OpenLLMetry trace failure in %s", origin, exc_info=exc)
    else:
        logger.warning(
            "OpenLLMetry failed to trace in %s, error: %s",
            origin,
            type(exc).__name__,
        )
        # Not exc_info: a formatted traceback ends with the message and quotes
        # each frame's source line, so it would reinstate at DEBUG exactly what
        # the WARNING above withholds.
        logger.debug(
            "OpenLLMetry trace failure in %s, error: %s\n%s",
            origin,
            type(exc).__name__,
            frames_without_content(exc),
        )
    if Config.exception_logger:
        try:
            Config.exception_logger(exc)
        except Exception:
            logger.debug(
                "OpenLLMetry exception_logger raised in %s", origin, exc_info=True
            )


@contextmanager
def guard_trace(logger: logging.Logger, origin: str):
    """Run instrumentation-only code, logging rather than raising its failures.

    The counterpart to ``dont_throw`` for a wrapper that must not guard the
    traced call itself: everything inside the block is this instrumentation's
    own work -- opening a span, setting attributes, ending it -- so a failure
    there is logged and the traced program carries on untouched. ``origin``
    names the site for the log line.

    Only ``Exception`` is swallowed: a cancellation has to keep unwinding.
    """
    try:
        yield
    except Exception as e:
        log_trace_failure(logger, origin, e)


@contextmanager
def guarded_span(logger: logging.Logger, origin: str, tracer, name: str, **kwargs):
    """Open a span whose own failures cannot reach the traced call.

    The span-lifecycle counterpart to ``guard_trace``, for a wrapper whose span
    surrounds the traced call rather than sitting beside it: there, neither
    opening nor closing the span may be allowed to reach the caller, because the
    call's own result or failure is the thing the caller is waiting for. A
    tracer that cannot produce a span yields ``INVALID_SPAN`` instead, so the
    body of the block needs no ``None`` check -- every ``set_attribute``,
    ``set_status`` and ``record_exception`` on it is a no-op.

    The span is closed with ``__exit__(None, None, None)``: callers on this path
    record failures themselves through ``record_error``, so the span must not
    also take a status from an exception on its way out.

    Only ``Exception`` is swallowed: a cancellation has to keep unwinding, and
    the span is still ended on its way past.
    """
    context_manager = None
    span = INVALID_SPAN
    try:
        context_manager = tracer.start_as_current_span(name, **kwargs)
        span = context_manager.__enter__()
    except Exception as e:
        log_trace_failure(logger, origin, e)
        context_manager = None
        span = INVALID_SPAN
    try:
        yield span
    finally:
        if context_manager is not None:
            with guard_trace(logger, f"{origin} end"):
                context_manager.__exit__(None, None, None)


def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.
    Works for both synchronous and asynchronous functions.

    Only sound when the decorated function is instrumentation code throughout: a
    wrapper that calls into the traced library must not decorate the call itself
    -- that would turn a library failure into a silent ``None``. Such wrappers
    use ``guard_trace`` around their span work instead (see the MCP client
    wrappers in ``instrumentation.py``).
    """
    logger = logging.getLogger(func.__module__)

    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            log_trace_failure(logger, func.__name__, e)

    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_trace_failure(logger, func.__name__, e)

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
