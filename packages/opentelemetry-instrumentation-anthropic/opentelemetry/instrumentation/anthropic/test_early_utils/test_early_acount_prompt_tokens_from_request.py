import asyncio
from unittest.mock import AsyncMock

import pytest
# Assuming the function is imported from the module
from opentelemetry.instrumentation.anthropic.utils import \
    acount_prompt_tokens_from_request


@pytest.mark.describe("acount_prompt_tokens_from_request")
class TestAcountPromptTokensFromRequest:

    @pytest.mark.happy_path
    @pytest.mark.asyncio
    async def test_single_prompt(self):
        """Test with a single prompt string to ensure correct token counting."""
        anthropic = AsyncMock()
        anthropic.count_tokens = AsyncMock(return_value=5)
        request = {"prompt": "This is a test prompt."}

        result = await acount_prompt_tokens_from_request(anthropic, request)

        assert result == 5
        anthropic.count_tokens.assert_awaited_once_with("This is a test prompt.")

    @pytest.mark.edge_case
    @pytest.mark.asyncio
    async def test_no_prompt_or_messages(self):
        """Test with no prompt or messages to ensure zero tokens are counted."""
        anthropic = AsyncMock()
        request = {}

        result = await acount_prompt_tokens_from_request(anthropic, request)

        assert result == 0
        anthropic.count_tokens.assert_not_awaited()

    @pytest.mark.edge_case
    @pytest.mark.asyncio
    async def test_message_with_non_string_content(self):
        """Test with message content that is not a string to ensure it is ignored."""
        anthropic = AsyncMock()
        anthropic.count_tokens = AsyncMock(return_value=0)
        request = {
            "messages": [
                {"content": 12345},  # Non-string content
                {"content": None}    # None content
            ]
        }

        result = await acount_prompt_tokens_from_request(anthropic, request)

        assert result == 0
        anthropic.count_tokens.assert_not_awaited()
