"""
Tests for the guardrails decorator functionality.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from traceloop.sdk.guardrails import (
    guardrails,
    get_current_score,
    InputExtractor,
)


def calculate_sentiment_score(event_data: dict) -> float:
    """Mock score calculator function."""
    return event_data.get("sentiment_score", 0.5)


@pytest.fixture
def mock_guardrails_client():
    """Mock guardrails client."""
    with patch("traceloop.sdk.guardrails.decorator._execute_guardrails") as mock:
        mock.return_value = {"sentiment_score": 0.8}
        yield mock


def test_guardrails_decorator_basic(mock_guardrails_client):
    """Test basic guardrails decorator functionality."""
    
    @guardrails(
        evaluator_slug="sentiment-analyzer",
        score_calculator=calculate_sentiment_score,
        input_schema={
            "text": InputExtractor(source="input", key="message"),
            "user_id": InputExtractor(source="input", key="user_id")
        }
    )
    def process_message(message: str, user_id: str) -> str:
        # Access the score within the function
        score = get_current_score()
        assert score == 0.8  # Should be set by the decorator
        
        if score < 0.3:
            return "REJECTED"
        elif score > 0.7:
            return "APPROVED"
        else:
            return "REVIEW"
    
    # Test the decorated function
    result = process_message("Hello world", "user123")
    assert result == "APPROVED"
    
    # Verify the mock was called
    mock_guardrails_client.assert_called_once()


def test_guardrails_decorator_with_regex():
    """Test guardrails decorator with regex pattern."""
    
    @guardrails(
        evaluator_slug="content-moderation",
        score_calculator=calculate_sentiment_score,
        input_schema={
            "email": InputExtractor(
                source="input", 
                key="contact_info",
                use_regex=True,
                regex_pattern=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
            )
        }
    )
    def process_contact(contact_info: str) -> str:
        score = get_current_score()
        return f"Processed with score: {score}"
    
    # This would need proper mocking of the regex extraction
    # For now, just test that the decorator doesn't crash
    with patch("traceloop.sdk.guardrails.decorator._execute_guardrails") as mock:
        mock.return_value = {"sentiment_score": 0.6}
        result = process_contact("Contact: john@example.com")
        assert "Processed with score: 0.6" in result


@pytest.mark.asyncio
async def test_guardrails_decorator_async():
    """Test guardrails decorator with async function."""
    
    @guardrails(
        evaluator_slug="async-analyzer",
        score_calculator=calculate_sentiment_score,
        input_schema={
            "text": InputExtractor(source="input", key="content")
        }
    )
    async def async_process(content: str) -> str:
        score = get_current_score()
        return f"Async processed with score: {score}"
    
    with patch("traceloop.sdk.guardrails.decorator._execute_guardrails") as mock:
        mock.return_value = {"sentiment_score": 0.9}
        result = await async_process("Async content")
        assert "Async processed with score: 0.9" in result


def test_input_extractor_creation():
    """Test InputExtractor dataclass creation."""
    extractor = InputExtractor(
        source="input",
        key="message",
        use_regex=False
    )
    
    assert extractor.source == "input"
    assert extractor.key == "message"
    assert extractor.use_regex is False
    assert extractor.regex_pattern is None


def test_get_current_score_outside_context():
    """Test get_current_score when no score is set."""
    # Should return None when no score is set in context
    score = get_current_score()
    assert score is None 