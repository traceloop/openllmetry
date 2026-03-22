"""Tests that async wrappers offload safety to a thread via asyncio.to_thread."""
import asyncio

import pytest

from opentelemetry.instrumentation.anthropic.safety import (
    _apply_completion_safety,
    _apply_prompt_safety,
)

pytestmark = [pytest.mark.fr, pytest.mark.asyncio]


async def test_apply_prompt_safety_via_to_thread():
    """Verify _apply_prompt_safety can run via asyncio.to_thread without blocking."""
    kwargs = {"messages": [{"role": "user", "content": "hello"}]}
    result = await asyncio.to_thread(_apply_prompt_safety, None, kwargs, "anthropic.chat")
    assert isinstance(result, dict)
    assert result.get("messages") is not None


async def test_apply_completion_safety_via_to_thread():
    """Verify _apply_completion_safety can run via asyncio.to_thread without blocking."""
    from types import SimpleNamespace
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text="reply")])
    # Should not raise
    await asyncio.to_thread(_apply_completion_safety, None, response, "anthropic.chat")
