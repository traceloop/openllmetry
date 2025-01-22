"""Test suite for Anthropic instrumentation."""

import pytest
from unittest.mock import Mock, patch

from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
from opentelemetry.instrumentation.anthropic.config import Config
from opentelemetry.trace import SpanKind
from opentelemetry.semconv_ai import SpanAttributes, LLMRequestTypeValues
from opentelemetry._events import Event

# Test fixtures
@pytest.fixture
def mock_anthropic():
    with patch("anthropic.Anthropic") as mock:
        yield mock

@pytest.fixture
def mock_tracer():
    mock = Mock()
    mock.start_as_current_span.return_value.__enter__.return_value = Mock()
    return mock

@pytest.fixture
def mock_event_logger():
    return Mock()

@pytest.fixture
def mock_meter():
    return Mock()

@pytest.fixture
def instrumentor(mock_tracer, mock_event_logger, mock_meter):
    instrumentor = AnthropicInstrumentor()
    instrumentor._tracer = mock_tracer
    instrumentor._event_logger = mock_event_logger
    instrumentor._meter = mock_meter
    return instrumentor

# Configuration Tests
def test_default_config():
    """Test default configuration values."""
    assert Config.use_legacy_attributes == True
    assert Config.capture_content == True

def test_config_override(instrumentor):
    """Test configuration override during instrumentation."""
    instrumentor._instrument(use_legacy_attributes=False)
    assert Config.use_legacy_attributes == False

# Event Emission Tests
def test_prompt_event_emission(instrumentor, mock_tracer, mock_event_logger):
    """Test prompt event emission in event-based mode."""
    # Configure for event-based mode
    Config.use_legacy_attributes = False
    
    # Mock span context
    span = mock_tracer.start_as_current_span.return_value.__enter__.return_value
    span.get_span_context.return_value = Mock(
        trace_id=123,
        span_id=456,
        trace_flags=1
    )
    
    # Test message creation
    kwargs = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "claude-2"
    }
    
    # Call the completion endpoint
    with patch("anthropic.resources.messages.Messages.create") as mock_create:
        mock_create.return_value = {
            "role": "assistant",
            "content": "Hi there!",
            "stop_reason": "end_turn"
        }
        
        instrumentor._instrument()
        mock_create(**kwargs)
        
        # Verify prompt event was emitted
        assert mock_event_logger.emit.call_count >= 1
        event = mock_event_logger.emit.call_args_list[0][0][0]
        assert isinstance(event, Event)
        assert event.name == "gen_ai.prompt"
        assert event.body["role"] == "user"
        assert event.body["content"] == "Hello"
        assert event.attributes["gen_ai.system"] == "anthropic"

def test_completion_event_emission(instrumentor, mock_tracer, mock_event_logger):
    """Test completion event emission in event-based mode."""
    # Configure for event-based mode
    Config.use_legacy_attributes = False
    
    # Mock span context
    span = mock_tracer.start_as_current_span.return_value.__enter__.return_value
    span.get_span_context.return_value = Mock(
        trace_id=123,
        span_id=456,
        trace_flags=1
    )
    
    # Test completion
    response = {
        "role": "assistant",
        "content": "Hi there!",
        "stop_reason": "end_turn",
        "model": "claude-2"
    }
    
    with patch("anthropic.resources.messages.Messages.create") as mock_create:
        mock_create.return_value = response
        
        instrumentor._instrument()
        mock_create(messages=[{"role": "user", "content": "Hello"}])
        
        # Verify completion event was emitted
        completion_event_calls = [
            call for call in mock_event_logger.emit.call_args_list 
            if call[0][0].name == "gen_ai.completion"
        ]
        assert len(completion_event_calls) >= 1
        event = completion_event_calls[0][0][0]
        assert isinstance(event, Event)
        assert event.name == "gen_ai.completion"
        assert event.body["role"] == "assistant"
        assert event.body["content"] == "Hi there!"
        assert event.body["finish_reason"] == "end_turn"

def test_tool_call_event_emission(instrumentor, mock_tracer, mock_event_logger):
    """Test tool call event emission in event-based mode."""
    # Configure for event-based mode
    Config.use_legacy_attributes = False
    
    # Mock span context
    span = mock_tracer.start_as_current_span.return_value.__enter__.return_value
    span.get_span_context.return_value = Mock(
        trace_id=123,
        span_id=456,
        trace_flags=1
    )
    
    # Test tool call
    kwargs = {
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [{
            "name": "calculator",
            "description": "Basic calculator",
            "input_schema": {"type": "object"}
        }],
        "model": "claude-2"
    }
    
    with patch("anthropic.resources.messages.Messages.create") as mock_create:
        mock_create.return_value = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "calc_1",
                    "name": "calculator",
                    "input": {"operation": "add", "numbers": [1, 2]}
                }
            ],
            "stop_reason": "end_turn"
        }
        
        instrumentor._instrument()
        mock_create(**kwargs)
        
        # Verify tool call events were emitted
        tool_event_calls = [
            call for call in mock_event_logger.emit.call_args_list 
            if call[0][0].name == "gen_ai.tool_call"
        ]
        assert len(tool_event_calls) >= 1
        event = tool_event_calls[0][0][0]
        assert isinstance(event, Event)
        assert event.name == "gen_ai.tool_call"
        assert event.body["name"] == "calculator"
        assert event.body["description"] == "Basic calculator" 

# Legacy Mode Tests
def test_legacy_mode_attributes(instrumentor, mock_tracer):
    """Test attribute-based tracking in legacy mode."""
    # Configure for legacy mode
    Config.use_legacy_attributes = True
    
    # Mock span
    span = mock_tracer.start_as_current_span.return_value.__enter__.return_value
    
    # Test message creation
    kwargs = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "claude-2"
    }
    
    with patch("anthropic.resources.messages.Messages.create") as mock_create:
        mock_create.return_value = {
            "role": "assistant",
            "content": "Hi there!",
            "stop_reason": "end_turn"
        }
        
        instrumentor._instrument()
        mock_create(**kwargs)
        
        # Verify span attributes were set
        span.set_attribute.assert_any_call(f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
        span.set_attribute.assert_any_call(f"{SpanAttributes.LLM_PROMPTS}.0.content", "Hello")
        span.set_attribute.assert_any_call(f"{SpanAttributes.LLM_COMPLETIONS}.0.content", "Hi there!")

def test_legacy_mode_with_system_message(instrumentor, mock_tracer):
    """Test legacy mode with system message."""
    Config.use_legacy_attributes = True
    
    span = mock_tracer.start_as_current_span.return_value.__enter__.return_value
    
    kwargs = {
        "system": "You are a helpful assistant",
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "claude-2"
    }
    
    with patch("anthropic.resources.messages.Messages.create") as mock_create:
        mock_create.return_value = {
            "role": "assistant",
            "content": "Hi there!",
            "stop_reason": "end_turn"
        }
        
        instrumentor._instrument()
        mock_create(**kwargs)
        
        # Verify system message was set
        span.set_attribute.assert_any_call(f"{SpanAttributes.LLM_PROMPTS}.0.role", "system")
        span.set_attribute.assert_any_call(f"{SpanAttributes.LLM_PROMPTS}.0.content", "You are a helpful assistant")

# Streaming Tests
def test_sync_streaming(instrumentor, mock_tracer, mock_event_logger):
    """Test synchronous streaming response handling."""
    Config.use_legacy_attributes = False
    
    span = mock_tracer.start_as_current_span.return_value.__enter__.return_value
    span.get_span_context.return_value = Mock(
        trace_id=123,
        span_id=456,
        trace_flags=1
    )
    
    class MockStream:
        def __iter__(self):
            yield Mock(
                type="message_start",
                message=Mock(model="claude-2", usage={"input_tokens": 10})
            )
            yield Mock(
                type="content_block_start",
                index=0
            )
            yield Mock(
                type="content_block_delta",
                index=0,
                delta=Mock(type="text_delta", text="Hi ")
            )
            yield Mock(
                type="content_block_delta",
                index=0,
                delta=Mock(type="text_delta", text="there!")
            )
            yield Mock(
                type="message_delta",
                delta=Mock(stop_reason="end_turn"),
                usage={"output_tokens": 5}
            )
    
    with patch("anthropic.resources.messages.Messages.create") as mock_create:
        mock_create.return_value = MockStream()
        
        instrumentor._instrument()
        response = mock_create(messages=[{"role": "user", "content": "Hello"}])
        
        # Consume the stream
        for _ in response:
            pass
        
        # Verify completion event was emitted with complete text
        completion_events = [
            call for call in mock_event_logger.emit.call_args_list 
            if call[0][0].name == "gen_ai.completion"
        ]
        assert len(completion_events) >= 1
        event = completion_events[-1][0][0]
        assert event.body["content"] == "Hi there!"
        assert event.body["finish_reason"] == "end_turn"

@pytest.mark.asyncio
async def test_async_streaming(instrumentor, mock_tracer, mock_event_logger):
    """Test asynchronous streaming response handling."""
    Config.use_legacy_attributes = False
    
    span = mock_tracer.start_as_current_span.return_value.__enter__.return_value
    span.get_span_context.return_value = Mock(
        trace_id=123,
        span_id=456,
        trace_flags=1
    )
    
    class MockAsyncStream:
        async def __aiter__(self):
            yield Mock(
                type="message_start",
                message=Mock(model="claude-2", usage={"input_tokens": 10})
            )
            yield Mock(
                type="content_block_start",
                index=0
            )
            yield Mock(
                type="content_block_delta",
                index=0,
                delta=Mock(type="text_delta", text="Hi ")
            )
            yield Mock(
                type="content_block_delta",
                index=0,
                delta=Mock(type="text_delta", text="there!")
            )
            yield Mock(
                type="message_delta",
                delta=Mock(stop_reason="end_turn"),
                usage={"output_tokens": 5}
            )
    
    with patch("anthropic.resources.messages.AsyncMessages.create") as mock_create:
        mock_create.return_value = MockAsyncStream()
        
        instrumentor._instrument()
        response = mock_create(messages=[{"role": "user", "content": "Hello"}])
        
        # Consume the stream
        async for _ in response:
            pass
        
        # Verify completion event was emitted with complete text
        completion_events = [
            call for call in mock_event_logger.emit.call_args_list 
            if call[0][0].name == "gen_ai.completion"
        ]
        assert len(completion_events) >= 1
        event = completion_events[-1][0][0]
        assert event.body["content"] == "Hi there!"
        assert event.body["finish_reason"] == "end_turn" 

# Error Handling Tests
def test_error_handling(instrumentor, mock_tracer):
    """Test error handling and span status."""
    span = mock_tracer.start_as_current_span.return_value.__enter__.return_value
    
    class MockError(Exception):
        pass
    
    with patch("anthropic.resources.messages.Messages.create") as mock_create:
        mock_create.side_effect = MockError("API Error")
        
        instrumentor._instrument()
        
        with pytest.raises(MockError):
            mock_create(messages=[{"role": "user", "content": "Hello"}])
        
        # Verify error was recorded
        span.set_status.assert_called_once()
        span.record_exception.assert_called_once()

def test_exception_logger(instrumentor, mock_tracer):
    """Test custom exception logger."""
    mock_exception_logger = Mock()
    Config.exception_logger = mock_exception_logger
    
    class MockError(Exception):
        pass
    
    with patch("anthropic.resources.messages.Messages.create") as mock_create:
        mock_create.side_effect = MockError("API Error")
        
        instrumentor._instrument()
        
        with pytest.raises(MockError):
            mock_create(messages=[{"role": "user", "content": "Hello"}])
        
        # Verify custom logger was called
        mock_exception_logger.assert_called_once()

def test_streaming_error_handling(instrumentor, mock_tracer):
    """Test error handling in streaming responses."""
    span = mock_tracer.start_as_current_span.return_value.__enter__.return_value
    
    class MockError(Exception):
        pass
    
    class ErrorStream:
        def __iter__(self):
            yield Mock(
                type="message_start",
                message=Mock(model="claude-2")
            )
            raise MockError("Stream Error")
    
    with patch("anthropic.resources.messages.Messages.create") as mock_create:
        mock_create.return_value = ErrorStream()
        
        instrumentor._instrument()
        response = mock_create(messages=[{"role": "user", "content": "Hello"}])
        
        with pytest.raises(MockError):
            for _ in response:
                pass
        
        # Verify error was recorded
        span.set_status.assert_called_once()
        span.record_exception.assert_called_once()

@pytest.mark.asyncio
async def test_async_streaming_error_handling(instrumentor, mock_tracer):
    """Test error handling in async streaming responses."""
    span = mock_tracer.start_as_current_span.return_value.__enter__.return_value
    
    class MockError(Exception):
        pass
    
    class AsyncErrorStream:
        async def __aiter__(self):
            yield Mock(
                type="message_start",
                message=Mock(model="claude-2")
            )
            raise MockError("Stream Error")
    
    with patch("anthropic.resources.messages.AsyncMessages.create") as mock_create:
        mock_create.return_value = AsyncErrorStream()
        
        instrumentor._instrument()
        response = mock_create(messages=[{"role": "user", "content": "Hello"}])
        
        with pytest.raises(MockError):
            async for _ in response:
                pass
        
        # Verify error was recorded
        span.set_status.assert_called_once()
        span.record_exception.assert_called_once() 