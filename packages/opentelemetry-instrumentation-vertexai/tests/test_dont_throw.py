import asyncio

import pytest
from opentelemetry.instrumentation.vertexai.utils import dont_throw


def test_dont_throw_swallows_sync_exceptions():
    @dont_throw
    def boom():
        raise RuntimeError("instrumentation failed")

    assert boom() is None, "sync instrumentation errors must not reach the caller"


@pytest.mark.asyncio
async def test_dont_throw_swallows_async_exceptions():
    """An async function returns a coroutine immediately, so a sync-only
    wrapper exits its try block before the body runs and the caller awaits
    outside the guard. span_utils.set_input_attributes and _handle_request are
    both async and both decorated, and _handle_request is awaited unguarded."""

    @dont_throw
    async def boom():
        raise RuntimeError("instrumentation failed")

    assert await boom() is None, "async instrumentation errors must not reach the caller"


@pytest.mark.asyncio
async def test_dont_throw_returns_async_values():
    @dont_throw
    async def fine():
        return "ok"

    assert await fine() == "ok", "decorator must not swallow the return value"


def test_dont_throw_preserves_sync_return():
    @dont_throw
    def fine():
        return "ok"

    assert fine() == "ok"


def test_dont_throw_picks_wrapper_by_function_kind():
    @dont_throw
    async def coro():
        return None

    @dont_throw
    def plain():
        return None

    assert asyncio.iscoroutinefunction(coro), "async functions need the async wrapper"
    assert not asyncio.iscoroutinefunction(plain)


def test_dont_throw_survives_a_failing_exception_logger():
    """A user-supplied exception_logger that itself raises must not reach the caller."""
    from opentelemetry.instrumentation.vertexai.config import Config

    def angry_logger(_e):
        raise ValueError("exception logger is broken")

    previous = Config.exception_logger
    Config.exception_logger = angry_logger
    try:
        @dont_throw
        def boom():
            raise RuntimeError("instrumentation failed")

        assert boom() is None, "a failing exception_logger must not surface to the caller"
    finally:
        Config.exception_logger = previous


@pytest.mark.asyncio
async def test_dont_throw_async_survives_a_failing_exception_logger():
    from opentelemetry.instrumentation.vertexai.config import Config

    def angry_logger(_e):
        raise ValueError("exception logger is broken")

    previous = Config.exception_logger
    Config.exception_logger = angry_logger
    try:
        @dont_throw
        async def boom():
            raise RuntimeError("instrumentation failed")

        assert await boom() is None
    finally:
        Config.exception_logger = previous
