"""Tests that async wrappers offload safety to a thread via asyncio.to_thread."""
import asyncio

import pytest

from opentelemetry.instrumentation.vertexai.safety import (
    _apply_completion_safety,
    _apply_prompt_safety,
)

pytestmark = [pytest.mark.fr, pytest.mark.asyncio]


async def test_apply_prompt_safety_via_to_thread():
    """Verify _apply_prompt_safety can run via asyncio.to_thread without blocking."""
    args = ("hello",)
    kwargs = {}
    result = await asyncio.to_thread(
        _apply_prompt_safety, None, args, kwargs, "vertexai.generate"
    )
    assert isinstance(result, tuple)
    assert len(result) == 2  # (args, kwargs)


async def test_apply_completion_safety_via_to_thread():
    """Verify _apply_completion_safety can run via asyncio.to_thread without blocking."""
    from types import SimpleNamespace
    response = SimpleNamespace(candidates=None)
    await asyncio.to_thread(
        _apply_completion_safety, None, response, "vertexai.generate"
    )
