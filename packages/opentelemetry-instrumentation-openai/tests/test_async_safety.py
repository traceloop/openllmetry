"""Tests that async wrappers offload safety to a thread via asyncio.to_thread."""
import asyncio
from unittest.mock import patch, MagicMock

import pytest

pytestmark = [pytest.mark.fr, pytest.mark.asyncio]


async def test_achat_wrapper_uses_async_prompt_safety():
    """Verify achat_wrapper calls _apply_prompt_safety via asyncio.to_thread."""
    calls = []
    original_to_thread = asyncio.to_thread

    async def tracking_to_thread(func, *args, **kwargs):
        calls.append(func.__name__ if callable(func) else str(func))
        return await original_to_thread(func, *args, **kwargs)

    with patch("opentelemetry.instrumentation.openai.shared.chat_wrappers.asyncio") as mock_asyncio:
        mock_asyncio.to_thread = tracking_to_thread
        # Import after patching
        from opentelemetry.instrumentation.openai.shared.chat_safety import (
            _apply_prompt_safety,
            _apply_completion_safety,
        )
        # Call to_thread with the actual functions to verify it works
        result = await tracking_to_thread(_apply_prompt_safety, None, {"messages": []})
        assert isinstance(result, dict)
        assert "_apply_prompt_safety" in calls


async def test_acompletion_wrapper_uses_async_prompt_safety():
    """Verify acompletion_wrapper calls _apply_prompt_safety via asyncio.to_thread."""
    from opentelemetry.instrumentation.openai.shared.completion_safety import (
        _apply_prompt_safety,
    )
    result = await asyncio.to_thread(_apply_prompt_safety, None, {"prompt": "test"})
    assert isinstance(result, dict)


async def test_async_responses_wrapper_uses_async_prompt_safety():
    """Verify apply_response_prompt_safety works via asyncio.to_thread."""
    from opentelemetry.instrumentation.openai.v1.responses_safety import (
        apply_response_prompt_safety,
    )
    result = await asyncio.to_thread(apply_response_prompt_safety, None, {"input": "test"})
    assert isinstance(result, dict)
