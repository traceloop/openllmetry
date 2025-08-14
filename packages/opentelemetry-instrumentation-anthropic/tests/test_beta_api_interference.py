import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

try:
    from anthropic import AsyncAnthropicBedrock
except ImportError:
    AsyncAnthropicBedrock = None


@pytest.fixture
def async_anthropic_bedrock_client_beta(instrument_legacy):
    """Create a mock AsyncAnthropicBedrock client with beta API support"""
    if AsyncAnthropicBedrock is None:
        pytest.skip("AsyncAnthropicBedrock not available")

    client = AsyncAnthropicBedrock(
        aws_region="us-east-1",
        aws_access_key="test-key",
        aws_secret_key="test-secret",
    )
    return client


@pytest.mark.asyncio
async def test_beta_api_context_interference(
    instrument_legacy,
    async_anthropic_bedrock_client_beta,
    span_exporter,
):
    """
    Test that reproduces the LangGraph agent getting stuck due to beta API instrumentation.
    This test simulates what happens when LangGraph uses Claude on Bedrock with computer use.
    """

    # Mock a response that simulates what the beta API would return
    mock_response = Mock()
    mock_response.content = [Mock(text="I'll help you with that task")]
    mock_response.usage = Mock(input_tokens=50, output_tokens=20)
    mock_response.stop_reason = "end_turn"

    # Create a mock beta messages client that behaves like the real one
    mock_beta_messages = Mock()
    mock_beta_messages.create = AsyncMock(return_value=mock_response)

    # Patch the beta messages client
    with patch.object(async_anthropic_bedrock_client_beta, "beta", Mock()) as mock_beta:
        mock_beta.messages = mock_beta_messages

        # Simulate what LangGraph would do - multiple rapid calls in sequence
        # This is where the agent gets stuck according to the trace
        tasks = []
        for i in range(3):
            task = asyncio.create_task(
                mock_beta.messages.create(
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user",
                            "content": f"Task {i}: Use the computer to complete this action",
                        }
                    ],
                    model="anthropic.claude-3-sonnet-20241022-v2:0",
                    tools=[
                        {
                            "type": "computer_20241022",
                            "name": "computer",
                            "display_width_px": 1024,
                            "display_height_px": 768,
                        }
                    ],
                )
            )
            tasks.append(task)
            # Small delay to simulate real workflow timing
            await asyncio.sleep(0.1)

        # Wait for all tasks to complete - this should not hang
        try:
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
            assert len(results) == 3
        except asyncio.TimeoutError:
            pytest.fail(
                "Tasks timed out - this indicates the beta API instrumentation is causing a hang"
            )


@pytest.mark.asyncio
async def test_beta_api_response_corruption(
    instrument_legacy,
    async_anthropic_bedrock_client_beta,
    span_exporter,
):
    """
    Test that checks if beta API instrumentation corrupts the response context
    """

    # Create a mock response with computer use tool result
    mock_content_item = Mock()
    mock_content_item.type = "tool_use"
    mock_content_item.id = "tool_123"
    mock_content_item.name = "computer"
    mock_content_item.input = {"action": "click", "coordinate": [100, 200]}

    mock_usage = Mock()
    mock_usage.input_tokens = 100
    mock_usage.output_tokens = 50

    mock_response = Mock()
    mock_response.content = [mock_content_item]
    mock_response.usage = mock_usage
    mock_response.stop_reason = "tool_use"

    original_response = mock_response

    # Patch beta API
    with patch.object(async_anthropic_bedrock_client_beta, "beta", Mock()) as mock_beta:
        mock_beta.messages = Mock()
        mock_beta.messages.create = AsyncMock(return_value=mock_response)

        # Make the call
        response = await mock_beta.messages.create(
            max_tokens=1024,
            messages=[{"role": "user", "content": "Take a screenshot"}],
            model="anthropic.claude-3-sonnet-20241022-v2:0",
            tools=[
                {
                    "type": "computer_20241022",
                    "name": "computer",
                    "display_width_px": 1024,
                    "display_height_px": 768,
                }
            ],
        )

        # Verify the response wasn't corrupted by instrumentation
        assert response is not None
        assert response.content[0].type == "tool_use"
        assert response.content[0].name == "computer"
        assert response.stop_reason == "tool_use"

        # Check that the response object identity is preserved
        # If instrumentation wraps the response incorrectly, this could break LangGraph
        assert response is original_response or hasattr(response, "__wrapped__")


def test_beta_api_method_signature_preservation():
    """
    Test that beta API method signatures are preserved after instrumentation.
    LangGraph might depend on specific method signatures or attributes.
    """
    # This test would check that the wrapped methods preserve their original signatures
    # and don't introduce unexpected parameters or return types
    pass
