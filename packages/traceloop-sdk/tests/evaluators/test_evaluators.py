import pytest
from unittest.mock import patch, MagicMock
from traceloop.sdk.evaluators import Evaluator


def test_evaluator_import():
    """Test that Evaluator can be imported successfully"""
    assert Evaluator is not None


def test_evaluator_run_method_exists():
    """Test that Evaluator.run method exists and is callable"""
    assert hasattr(Evaluator, 'run')
    assert callable(getattr(Evaluator, 'run'))


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_evaluator_run_success():
    """Test successful evaluator execution with mocked API and SSE calls"""
    
    # Mock HTTP client response
    mock_http_response = {
        "execution_id": "test-execution-123",
        "stream_url": "/stream/test-execution-123"
    }
    
    # Mock SSE result
    mock_sse_result = {
        "result": "positive",
        "score": 0.85,
        "explanation": "The input shows positive sentiment"
    }
    
    with patch.object(Evaluator, '_get_http_client_static') as mock_get_client, \
         patch.object(Evaluator, '_wait_for_sse_result') as mock_wait_sse:
        
        # Setup HTTP client mock
        mock_client = MagicMock()
        mock_client.post.return_value = mock_http_response
        mock_get_client.return_value = mock_client
        
        # Setup SSE wait mock
        mock_wait_sse.return_value = mock_sse_result
        
        # Execute the evaluator
        result = Evaluator.run(
            evaluator_slug="test-evaluator",
            input={
                "text": "I love this product!",
                "context": "product review"
            },
            timeout_in_sec=60
        )
        
        # Verify the result
        assert result == mock_sse_result
        assert result["result"] == "positive"
        assert result["score"] == 0.85
        
        # Verify HTTP client was called correctly
        mock_get_client.assert_called_once()
        mock_client.post.assert_called_once()
        
        # Get the actual call arguments
        call_args = mock_client.post.call_args
        endpoint = call_args[0][0]
        body = call_args[0][1]
        
        assert endpoint == "evaluators/slug/test-evaluator/execute"
        assert "input_schema_mapping" in body
        assert body["source"] == "experiments"
        
        # Verify SSE wait was called correctly
        mock_wait_sse.assert_called_once_with(
            "/stream/test-execution-123",
            "test-execution-123",
            60
        )


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_evaluator_run_api_failure():
    """Test evaluator execution when API call fails"""
    
    with patch.object(Evaluator, '_get_http_client_static') as mock_get_client:
        # Setup HTTP client to return None (failure)
        mock_client = MagicMock()
        mock_client.post.return_value = None
        mock_get_client.return_value = mock_client
        
        # Execute the evaluator and expect exception
        with pytest.raises(Exception, match="Failed to execute evaluator test-evaluator"):
            Evaluator.run(
                evaluator_slug="test-evaluator",
                input={"text": "test input"},
                timeout_in_sec=60
            )
        
        # Verify HTTP client was called
        mock_get_client.assert_called_once()
        mock_client.post.assert_called_once()


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_evaluator_run_sse_timeout():
    """Test evaluator execution when SSE stream times out"""
    
    # Mock HTTP client response
    mock_http_response = {
        "execution_id": "test-execution-456",
        "stream_url": "/stream/test-execution-456"
    }
    
    with patch.object(Evaluator, '_get_http_client_static') as mock_get_client, \
         patch.object(Evaluator, '_wait_for_sse_result') as mock_wait_sse:
        
        # Setup HTTP client mock
        mock_client = MagicMock()
        mock_client.post.return_value = mock_http_response
        mock_get_client.return_value = mock_client
        
        # Setup SSE wait to raise timeout
        mock_wait_sse.side_effect = TimeoutError("Evaluator execution test-execution-456 timed out after 30s")
        
        # Execute the evaluator and expect timeout
        with pytest.raises(TimeoutError, match="timed out after 30s"):
            Evaluator.run(
                evaluator_slug="test-evaluator",
                input={"text": "test input"},
                timeout_in_sec=30
            )
        
        # Verify both HTTP and SSE calls were made
        mock_get_client.assert_called_once()
        mock_client.post.assert_called_once()
        mock_wait_sse.assert_called_once()


def test_evaluator_run_missing_api_key():
    """Test evaluator execution when API key is missing"""
    
    with patch.dict("os.environ", {}, clear=True):
        # Execute the evaluator and expect exception
        with pytest.raises(ValueError, match="TRACELOOP_API_KEY environment variable is required"):
            Evaluator.run(
                evaluator_slug="test-evaluator",
                input={"text": "test input"}
            )

    """Test that input is correctly transformed to schema mapping"""
    
    # Mock HTTP client response
    mock_http_response = {
        "execution_id": "test-execution-789",
        "stream_url": "/stream/test-execution-789"
    }
    
    mock_sse_result = {"result": "processed"}
    
    with patch.object(Evaluator, '_get_http_client_static') as mock_get_client, \
         patch.object(Evaluator, '_wait_for_sse_result') as mock_wait_sse:
        
        # Setup mocks
        mock_client = MagicMock()
        mock_client.post.return_value = mock_http_response
        mock_get_client.return_value = mock_client
        mock_wait_sse.return_value = mock_sse_result
        
        # Execute with complex input
        test_input = {
            "user_query": "What is machine learning?",
            "context": "educational content",
            "difficulty": "beginner"
        }
        
        result = Evaluator.run(
            evaluator_slug="education-evaluator",
            input=test_input,
            timeout_in_sec=90
        )
        
        # Verify the result
        assert result == mock_sse_result
        
        # Verify the input schema mapping was created correctly
        call_args = mock_client.post.call_args
        body = call_args[0][1]
        
        assert "input_schema_mapping" in body
        input_mapping = body["input_schema_mapping"]
        
        # Check that all input fields are mapped directly
        assert "user_query" in input_mapping
        assert "context" in input_mapping
        assert "difficulty" in input_mapping
        
        # Check that each field has the correct source
        for field_name, field_value in test_input.items():
            assert input_mapping[field_name]["source"] == field_value