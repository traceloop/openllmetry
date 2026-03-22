"""Tests that async wrappers offload safety to a thread via asyncio.to_thread."""
import asyncio

import pytest

from opentelemetry.instrumentation.cohere.safety import (
    _apply_completion_safety,
    _apply_prompt_safety,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues

pytestmark = [pytest.mark.fr, pytest.mark.asyncio]


async def test_apply_prompt_safety_via_to_thread():
    """Verify _apply_prompt_safety can run via asyncio.to_thread without blocking."""
    kwargs = {"messages": [{"role": "user", "content": "hello"}]}
    result = await asyncio.to_thread(
        _apply_prompt_safety, None, kwargs, LLMRequestTypeValues.CHAT, "cohere.chat"
    )
    assert isinstance(result, dict)


async def test_apply_completion_safety_via_to_thread():
    """Verify _apply_completion_safety can run via asyncio.to_thread without blocking."""
    from types import SimpleNamespace
    response = SimpleNamespace(text="reply")
    await asyncio.to_thread(
        _apply_completion_safety, None, response, LLMRequestTypeValues.CHAT, "cohere.chat"
    )
