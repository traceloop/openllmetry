"""Tests that async wrappers offload safety to a thread via asyncio.to_thread."""
import asyncio

import pytest

from opentelemetry.instrumentation.litellm.safety import (
    apply_completion_safety,
    apply_prompt_safety,
)

pytestmark = [pytest.mark.fr, pytest.mark.asyncio]


async def test_apply_prompt_safety_via_to_thread():
    """Verify apply_prompt_safety can run via asyncio.to_thread without blocking."""
    args = ("model",)
    kwargs = {"messages": [{"role": "user", "content": "hello"}]}
    result = await asyncio.to_thread(
        apply_prompt_safety, None, args, kwargs, "chat", "litellm.completion"
    )
    assert isinstance(result, tuple)
    assert len(result) == 2  # (args, kwargs)


async def test_apply_completion_safety_via_to_thread():
    """Verify apply_completion_safety can run via asyncio.to_thread without blocking."""
    from types import SimpleNamespace
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="reply"), text="reply")])
    await asyncio.to_thread(
        apply_completion_safety, None, response, "chat", "litellm.completion"
    )
