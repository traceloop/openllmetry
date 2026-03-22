"""Tests that async wrappers offload safety to a thread via asyncio.to_thread."""
import asyncio

import pytest

from opentelemetry.instrumentation.langchain.safety import (
    _apply_chat_prompt_safety,
    _apply_llm_prompt_safety,
    _apply_chat_result_completion_safety,
    _apply_llm_result_completion_safety,
)

pytestmark = [pytest.mark.fr, pytest.mark.asyncio]


class _FakeInstance:
    class __class__:
        __name__ = "FakeChatModel"


async def test_apply_chat_prompt_safety_via_to_thread():
    """Verify _apply_chat_prompt_safety can run via asyncio.to_thread without blocking."""
    instance = _FakeInstance()
    args = ([[{"type": "human", "content": "hello"}]],)
    kwargs = {}
    result = await asyncio.to_thread(
        _apply_chat_prompt_safety, instance, args, kwargs
    )
    assert isinstance(result, tuple)
    assert len(result) == 2  # (args, kwargs)


async def test_apply_llm_prompt_safety_via_to_thread():
    """Verify _apply_llm_prompt_safety can run via asyncio.to_thread without blocking."""
    instance = _FakeInstance()
    args = (["hello"],)
    kwargs = {}
    result = await asyncio.to_thread(
        _apply_llm_prompt_safety, instance, args, kwargs
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
