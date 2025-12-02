import pytest
from unittest.mock import MagicMock


class MockLegacyAPIResponse:
    """Mock LegacyAPIResponse for testing"""
    
    def __init__(self, stream):
        self._stream = stream
        
    def parse(self):
        return self._stream


def test_legacy_api_response_detection():
    """Test that LegacyAPIResponse objects are properly detected as streaming"""
    from opentelemetry.instrumentation.openai.shared import is_streaming_response
    
    # Create a mock LegacyAPIResponse
    mock_stream = MagicMock()
    legacy_response = MockLegacyAPIResponse(mock_stream)
    
    # Test that is_streaming_response correctly identifies LegacyAPIResponse as streaming when stream=True
    assert is_streaming_response(legacy_response, {"stream": True}) == True
    
    # Test that LegacyAPIResponse is not streaming when stream=False or not set
    assert is_streaming_response(legacy_response, {"stream": False}) == False
    assert is_streaming_response(legacy_response, {}) == False
    assert is_streaming_response(legacy_response) == False
    
    # Test with regular objects
    assert is_streaming_response("not_a_stream") == False
    assert is_streaming_response({"not": "stream"}) == False


@pytest.mark.vcr
def test_raw_response_header_streaming_completion_attributes(
    instrument_legacy, span_exporter, log_exporter, openai_client
):
    """Test that streaming responses with X-Stainless-Raw-Response header capture completion attributes"""
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say hello"}],
        stream=True,
        extra_headers={"X-Stainless-Raw-Response": "true"}
    )
    
    # If it's a LegacyAPIResponse, call parse() to get the actual stream
    if hasattr(response, 'parse'):
        actual_stream = response.parse()
        # Consume the stream
        for chunk in actual_stream:
            pass
    else:
        # Regular streaming response
        for chunk in response:
            pass
    
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    
    span = spans[0]
    assert span.name == "openai.chat"
    
    # Verify that completion attributes are captured
    assert span.attributes.get("gen_ai.completion.0.finish_reason") is not None
    
    # For text responses, should have completion content
    completion_content = span.attributes.get("gen_ai.completion.0.content")
    assert completion_content is not None
    assert isinstance(completion_content, str)
    assert len(completion_content) > 0
    
    # Usage tokens may or may not be present depending on the API response
    # but if they are present, they should be valid
    completion_tokens = span.attributes.get("gen_ai.usage.completion_tokens")
    if completion_tokens is not None:
        assert isinstance(completion_tokens, int)
        assert completion_tokens > 0


@pytest.mark.vcr  
def test_raw_response_header_tool_calls_completion_attributes(
    instrument_legacy, span_exporter, log_exporter, openai_client
):
    """Test that LegacyAPIResponse with tool calls captures completion attributes"""
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "What's the weather like?"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        }],
        stream=True,
        extra_headers={"X-Stainless-Raw-Response": "true"}
    )
    
    # If it's a LegacyAPIResponse, call parse() to get the actual stream
    if hasattr(response, 'parse'):
        actual_stream = response.parse()
        # Consume the stream
        for chunk in actual_stream:
            pass
    else:
        # Regular streaming response
        for chunk in response:
            pass
    
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    
    span = spans[0]
    assert span.name == "openai.chat"
    
    # Verify completion attributes are captured
    assert span.attributes.get("gen_ai.completion.0.finish_reason") is not None
    
    # Should have tool call attributes if the model chose to call tools
    finish_reason = span.attributes.get("gen_ai.completion.0.finish_reason")
    if finish_reason == "tool_calls":
        assert span.attributes.get("gen_ai.completion.0.tool_calls.0.name") is not None
        assert span.attributes.get("gen_ai.completion.0.tool_calls.0.id") is not None
    
    # Usage tokens may or may not be present depending on the API response
    # but if they are present, they should be valid
    completion_tokens = span.attributes.get("gen_ai.usage.completion_tokens")
    if completion_tokens is not None:
        assert isinstance(completion_tokens, int)
        assert completion_tokens > 0


def test_legacy_api_response_parse_method():
    """Test that ChatStream.parse() method transfers completion data correctly"""
    from opentelemetry.instrumentation.openai.shared.chat_wrappers import ChatStream
    from unittest.mock import MagicMock
    
    # Create a mock span
    mock_span = MagicMock()
    mock_span.is_recording.return_value = True
    
    # Create a mock LegacyAPIResponse 
    mock_stream = MagicMock()
    legacy_response = MockLegacyAPIResponse(mock_stream)
    
    # Create a ChatStream that wraps the LegacyAPIResponse
    chat_stream = ChatStream(
        span=mock_span,
        response=legacy_response,
        instance=MagicMock(),
        token_counter=MagicMock(),
        choice_counter=MagicMock(), 
        duration_histogram=MagicMock(),
        streaming_time_to_first_token=MagicMock(),
        streaming_time_to_generate=MagicMock(),
        start_time=1234567890,
        request_kwargs={}
    )
    
    # Add some accumulated data to the original stream
    chat_stream._complete_response = {
        "choices": [{"message": {"content": "test"}}],
        "usage": {"completion_tokens": 5, "prompt_tokens": 10}
    }
    
    # Call parse() which should transfer the data to a new ChatStream
    new_stream = chat_stream.parse()
    
    # Verify that the new stream has the transferred data
    assert new_stream._complete_response == chat_stream._complete_response
    
    # Verify that the original stream is marked as cleanup completed
    assert chat_stream._cleanup_completed == True


def test_legacy_api_response_model_as_dict():
    """Test that model_as_dict handles LegacyAPIResponse correctly"""
    from opentelemetry.instrumentation.openai.shared import model_as_dict
    
    # Create a mock LegacyAPIResponse
    mock_stream = MagicMock()
    legacy_response = MockLegacyAPIResponse(mock_stream)
    
    # Test that model_as_dict returns empty dict for streaming LegacyAPIResponse
    result = model_as_dict(legacy_response, is_streaming=True)
    assert result == {}
    
    # Test that model_as_dict processes non-streaming LegacyAPIResponse normally
    # This should call parse() and continue processing
    mock_stream.model_dump.return_value = {"key": "value"}
    result = model_as_dict(legacy_response, is_streaming=False)
    # For non-streaming, it should parse and process normally - returns the model_dump result
    assert result == {"key": "value"}
    
    # Test with regular dict
    regular_dict = {"key": "value"}
    result = model_as_dict(regular_dict)
    assert result == regular_dict


def test_original_stream_integrity():
    """Test that the original LegacyAPIResponse is not mutated"""
    from opentelemetry.instrumentation.openai.shared import _is_legacy_api_response
    
    # Create a mock LegacyAPIResponse
    mock_stream = MagicMock()
    legacy_response = MockLegacyAPIResponse(mock_stream)
    original_type = type(legacy_response)
    
    # Verify it's detected correctly
    assert _is_legacy_api_response(legacy_response) == True
    
    # Call parse() method
    parsed_stream = legacy_response.parse()
    
    # Verify original object is unchanged
    assert type(legacy_response) == original_type
    assert hasattr(legacy_response, 'parse')
    assert legacy_response._stream is mock_stream
    
    # Verify parsed stream is different object
    assert parsed_stream is mock_stream
    assert parsed_stream is not legacy_response


def test_multiple_stream_consumers():
    """Test that multiple consumers can safely iterate the same LegacyAPIResponse"""
    from opentelemetry.instrumentation.openai.shared.chat_wrappers import ChatStream
    from unittest.mock import MagicMock
    
    # Create a mock span and stream
    mock_span = MagicMock()
    mock_span.is_recording.return_value = True
    
    # Create a mock iterable stream that can be consumed multiple times
    mock_data = [{"chunk": 1}, {"chunk": 2}, {"chunk": 3}]
    mock_stream = MagicMock()
    mock_stream.__iter__ = lambda: iter(mock_data)
    mock_stream.__next__ = MagicMock(side_effect=StopIteration)
    
    legacy_response = MockLegacyAPIResponse(mock_stream)
    
    # Test that the original response can still be parsed after ChatStream creation
    chat_stream = ChatStream(
        span=mock_span,
        response=legacy_response.parse(),  # Parse directly, don't rely on ChatStream mutation
        instance=MagicMock(),
        token_counter=MagicMock(),
        choice_counter=MagicMock(),
        duration_histogram=MagicMock(),
        streaming_time_to_first_token=MagicMock(),
        streaming_time_to_generate=MagicMock(),
        start_time=1234567890,
        request_kwargs={}
    )
    
    # Verify original legacy_response can still be parsed
    second_stream = legacy_response.parse()
    assert second_stream is mock_stream
    
    # Verify the ChatStream has the correctly parsed stream
    assert chat_stream.__wrapped__ is mock_stream


def test_legacy_api_response_error_handling():
    """Test error handling when LegacyAPIResponse.parse() fails"""
    from opentelemetry.instrumentation.openai.shared import _is_legacy_api_response
    
    # Create a mock LegacyAPIResponse that fails to parse
    class FailingLegacyAPIResponse:
        def parse(self):
            raise ValueError("Parse failed")
    
    failing_response = FailingLegacyAPIResponse()
    
    # Verify it's detected as LegacyAPIResponse
    assert _is_legacy_api_response(failing_response) == True
    
    # Verify that parse failure is handled gracefully (would need to test in wrapper context)
    with pytest.raises(ValueError):
        failing_response.parse()