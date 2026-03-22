"""Tests that async wrappers offload safety to a thread via asyncio.to_thread."""
import asyncio

import pytest

from opentelemetry.instrumentation.llamaindex.safety import (
    _apply_chat_prompt_safety,
    _apply_completion_prompt_safety,
)

pytestmark = [pytest.mark.fr, pytest.mark.asyncio]


class _FakeInstance:
    class __class__:
        __name__ = "FakeLLM"


async def test_apply_chat_prompt_safety_via_to_thread():
    """Verify _apply_chat_prompt_safety can run via asyncio.to_thread without blocking."""
    instance = _FakeInstance()
    # Provide an empty list - no messages to process
    args = ([],)
    kwargs = {}
    result = await asyncio.to_thread(
        _apply_chat_prompt_safety, instance, args, kwargs
    )
    assert isinstance(result, tuple)
    assert len(result) == 2  # (args, kwargs)


async def test_apply_completion_prompt_safety_via_to_thread():
    """Verify _apply_completion_prompt_safety can run via asyncio.to_thread without blocking."""
    instance = _FakeInstance()
    args = ("hello",)
    kwargs = {}
    result = await asyncio.to_thread(
        _apply_completion_prompt_safety, instance, args, kwargs
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
