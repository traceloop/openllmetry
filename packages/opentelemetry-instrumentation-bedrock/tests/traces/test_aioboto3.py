"""Tests for aioboto3 async client instrumentation"""
import asyncio
import os
from unittest.mock import AsyncMock, patch

import aioboto3
import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


@pytest.mark.asyncio
async def test_aioboto3_converse(instrument_legacy, span_exporter, log_exporter):
    """Test that aioboto3 async clients are properly instrumented"""

    session = aioboto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
        region_name="us-east-1",
    )

    # Mock response
    mock_response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Test response from aioboto3"}]
            }
        },
        "usage": {
            "inputTokens": 10,
            "outputTokens": 5,
            "totalTokens": 15
        },
        "stopReason": "end_turn"
    }

    async with session.client("bedrock-runtime") as client:
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": "Tell me a joke about opentelemetry"},
                ],
            }
        ]

        with patch.object(client, '_make_api_call', new=AsyncMock(return_value=mock_response)):
            response = await client.converse(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                messages=messages,
                inferenceConfig={"temperature": 0.5},
            )

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    assert spans[0].name == "bedrock.converse"

    span = spans[0]

    # Assert on model and system
    assert span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "claude-3-haiku-20240307-v1:0"

    # Assert on usage
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 5


@pytest.mark.asyncio
async def test_aioboto3_wrapping(instrument_legacy):
    """Test that aioboto3 client methods are wrapped"""

    session = aioboto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
        region_name="us-east-1",
    )

    async with session.client("bedrock-runtime") as client:
        # Verify methods are wrapped
        assert hasattr(client.converse, '__wrapped__')
        assert hasattr(client.converse_stream, '__wrapped__')
        assert hasattr(client.invoke_model, '__wrapped__')
        assert hasattr(client.invoke_model_with_response_stream, '__wrapped__')
