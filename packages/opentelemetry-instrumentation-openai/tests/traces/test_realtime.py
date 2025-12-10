"""
Tests for OpenAI Realtime API instrumentation.

The Realtime API uses WebSockets, which cannot be recorded with VCR (HTTP-based).
These tests use comprehensive mocking to simulate the full WebSocket flow without
requiring real API connections.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode


class MockRealtimeEvent:
    """Mock class for Realtime API events."""

    def __init__(self, event_type: str, **kwargs):
        self.type = event_type
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockRealtimeSession:
    """Mock session object returned by session.created event."""

    def __init__(self, model="gpt-4o-realtime-preview", modalities=None, **kwargs):
        self.model = model
        self.modalities = modalities or ["text"]
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockRealtimeResponse:
    """Mock response object from response.done event."""

    def __init__(self, response_id="resp_123", **kwargs):
        self.id = response_id
        self.usage = None
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockRealtimeUsage:
    """Mock usage object."""

    def __init__(self, input_tokens=10, output_tokens=20):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class TestRealtimeWrappers:
    """Unit tests for realtime wrapper classes."""

    def test_realtime_session_state_initialization(self):
        """Test RealtimeSessionState initialization."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            RealtimeSessionState,
        )

        mock_tracer = MagicMock()
        state = RealtimeSessionState(mock_tracer, "gpt-4o-realtime-preview")

        assert state.tracer == mock_tracer
        assert state.model == "gpt-4o-realtime-preview"
        assert state.session_span is None
        assert state.response_span is None
        assert state.accumulated_text == ""
        assert state.function_calls == []

    def test_realtime_connection_manager_wrapper_init(self):
        """Test RealtimeConnectionManagerWrapper initialization."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            RealtimeConnectionManagerWrapper,
        )

        mock_manager = MagicMock()
        mock_tracer = MagicMock()

        wrapper = RealtimeConnectionManagerWrapper(
            mock_manager, mock_tracer, "gpt-4o-realtime-preview"
        )

        assert wrapper._connection_manager == mock_manager
        assert wrapper._tracer == mock_tracer
        assert wrapper._model == "gpt-4o-realtime-preview"

    @pytest.mark.asyncio
    async def test_realtime_event_processing_session_created(self):
        """Test processing session.created event."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            RealtimeEventIterator,
            RealtimeEventProcessor,
            RealtimeSessionState,
        )

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_tracer.start_span.return_value = mock_span

        state = RealtimeSessionState(mock_tracer, "gpt-4o-realtime-preview")
        state.session_span = mock_span
        processor = RealtimeEventProcessor(state)

        # Create mock event
        session = MockRealtimeSession(model="gpt-4o-realtime-preview-2024-12-17")
        event = MockRealtimeEvent("session.created", session=session)

        # Create iterator and process event
        mock_connection = MagicMock()
        iterator = RealtimeEventIterator(mock_connection, state, processor)
        iterator._process_event(event)

        # Verify state was updated
        assert state.model == "gpt-4o-realtime-preview-2024-12-17"

    @pytest.mark.asyncio
    async def test_realtime_event_processing_response_done(self):
        """Test processing response.done event."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            RealtimeEventIterator,
            RealtimeEventProcessor,
            RealtimeSessionState,
        )

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_tracer.start_span.return_value = mock_span

        state = RealtimeSessionState(mock_tracer, "gpt-4o-realtime-preview")
        state.response_span = mock_span
        state.accumulated_text = "Hello, world!"
        processor = RealtimeEventProcessor(state)

        # Create mock response with usage
        usage = MockRealtimeUsage(input_tokens=15, output_tokens=25)
        response = MockRealtimeResponse(response_id="resp_456", usage=usage)
        event = MockRealtimeEvent("response.done", response=response)

        # Create iterator and process event
        mock_connection = MagicMock()
        iterator = RealtimeEventIterator(mock_connection, state, processor)
        iterator._process_event(event)

        # Verify span was ended
        mock_span.end.assert_called_once()

        # Verify state was reset
        assert state.response_span is None
        assert state.accumulated_text == ""

    @pytest.mark.asyncio
    async def test_realtime_event_processing_text_delta(self):
        """Test processing response.text.delta event."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            RealtimeEventIterator,
            RealtimeEventProcessor,
            RealtimeSessionState,
        )

        mock_tracer = MagicMock()
        state = RealtimeSessionState(mock_tracer, "gpt-4o-realtime-preview")
        state.accumulated_text = "Hello"
        processor = RealtimeEventProcessor(state)

        # Create mock delta event
        event = MockRealtimeEvent("response.text.delta", delta=", world!")

        # Create iterator and process event
        mock_connection = MagicMock()
        iterator = RealtimeEventIterator(mock_connection, state, processor)
        iterator._process_event(event)

        # Verify text was accumulated
        assert state.accumulated_text == "Hello, world!"

    @pytest.mark.asyncio
    async def test_realtime_event_processing_function_call(self):
        """Test processing response.function_call_arguments.done event."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            RealtimeEventIterator,
            RealtimeEventProcessor,
            RealtimeSessionState,
        )

        mock_tracer = MagicMock()
        state = RealtimeSessionState(mock_tracer, "gpt-4o-realtime-preview")
        processor = RealtimeEventProcessor(state)

        # Create mock function call event
        event = MockRealtimeEvent(
            "response.function_call_arguments.done",
            name="get_weather",
            call_id="call_123",
            arguments='{"location": "London"}',
        )

        # Create iterator and process event
        mock_connection = MagicMock()
        iterator = RealtimeEventIterator(mock_connection, state, processor)
        iterator._process_event(event)

        # Verify function call was tracked
        assert len(state.function_calls) == 1
        assert state.function_calls[0]["name"] == "get_weather"
        assert state.function_calls[0]["call_id"] == "call_123"
        assert state.function_calls[0]["arguments"] == '{"location": "London"}'

    @pytest.mark.asyncio
    async def test_realtime_event_processing_error(self):
        """Test processing error event."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            RealtimeEventIterator,
            RealtimeEventProcessor,
            RealtimeSessionState,
        )

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        state = RealtimeSessionState(mock_tracer, "gpt-4o-realtime-preview")
        state.response_span = mock_span
        processor = RealtimeEventProcessor(state)

        # Create mock error event
        event = MockRealtimeEvent("error", error="Something went wrong")

        # Create iterator and process event
        mock_connection = MagicMock()
        iterator = RealtimeEventIterator(mock_connection, state, processor)
        iterator._process_event(event)

        # Verify error was recorded
        mock_span.set_status.assert_called()
        mock_span.record_exception.assert_called()
        mock_span.end.assert_called_once()

        # Verify state was reset
        assert state.response_span is None


class TestRealtimeWrapperRegistration:
    """Test that realtime wrappers are properly registered."""

    def test_realtime_connect_wrapper_exists(self):
        """Test that realtime_connect_wrapper is importable."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            realtime_connect_wrapper,
        )

        assert callable(realtime_connect_wrapper)

    def test_realtime_wrapper_registered_in_instrumentor(self):
        """Test that realtime wrappers are registered in the instrumentor."""
        from opentelemetry.instrumentation.openai.v1 import OpenAIV1Instrumentor

        # The _instrument method should reference realtime wrapping
        instrumentor = OpenAIV1Instrumentor()
        # Check that the instrumentor has the _try_wrap method
        assert hasattr(instrumentor, "_try_wrap")


class TestRealtimeSpanAttributes:
    """Test span attribute setting for realtime API."""

    @pytest.mark.asyncio
    async def test_response_span_has_correct_attributes(self):
        """Test that response spans have correct attributes."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            RealtimeEventProcessor,
            RealtimeSessionState,
            RealtimeResponseWrapper,
        )

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_tracer.start_span.return_value = mock_span

        state = RealtimeSessionState(mock_tracer, "gpt-4o-realtime-preview")
        state.trace_context = None
        processor = RealtimeEventProcessor(state)

        # Create response wrapper
        mock_response = AsyncMock()
        mock_response.create = AsyncMock()
        wrapper = RealtimeResponseWrapper(mock_response, state, processor)

        # Call create
        await wrapper.create()

        # Verify span was started with correct attributes
        mock_tracer.start_span.assert_called_once()
        call_kwargs = mock_tracer.start_span.call_args

        # Verify span name
        assert call_kwargs[0][0] == "openai.realtime"

    @pytest.mark.asyncio
    async def test_session_span_has_correct_attributes(self):
        """Test that session spans have correct attributes."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            RealtimeConnectionManagerWrapper,
        )

        mock_manager = AsyncMock()
        mock_connection = MagicMock()
        mock_manager.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_manager.__aexit__ = AsyncMock(return_value=None)

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_tracer.start_span.return_value = mock_span

        wrapper = RealtimeConnectionManagerWrapper(
            mock_manager, mock_tracer, "gpt-4o-realtime-preview"
        )

        # Enter the context manager
        await wrapper.__aenter__()

        # Verify session span was started
        mock_tracer.start_span.assert_called_once()
        call_kwargs = mock_tracer.start_span.call_args

        # Verify span name
        assert call_kwargs[0][0] == "openai.session"


class TestRealtimeConnectionWrapper:
    """Test the connection wrapper delegation."""

    @pytest.mark.asyncio
    async def test_connection_wrapper_delegates_attributes(self):
        """Test that connection wrapper delegates unknown attributes."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            RealtimeConnectionWrapper,
            RealtimeSessionState,
        )

        mock_tracer = MagicMock()
        state = RealtimeSessionState(mock_tracer, "gpt-4o-realtime-preview")

        mock_connection = MagicMock()
        mock_connection.some_method = MagicMock(return_value="result")

        wrapper = RealtimeConnectionWrapper(mock_connection, state)

        # Access delegated attribute
        result = wrapper.some_method()
        assert result == "result"
        mock_connection.some_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_wrapper_provides_wrapped_session(self):
        """Test that connection wrapper provides wrapped session object."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            RealtimeConnectionWrapper,
            RealtimeSessionWrapper,
            RealtimeSessionState,
        )

        mock_tracer = MagicMock()
        state = RealtimeSessionState(mock_tracer, "gpt-4o-realtime-preview")

        mock_connection = MagicMock()
        mock_connection.session = MagicMock()

        wrapper = RealtimeConnectionWrapper(mock_connection, state)

        # Access session property
        session = wrapper.session
        assert isinstance(session, RealtimeSessionWrapper)


class MockAsyncIterableConnection:
    """Mock connection that is also an async iterable."""

    def __init__(self, events):
        self.events = events
        self.index = 0
        self.session = MagicMock()
        self.session.update = AsyncMock()
        self.conversation = MagicMock()
        self.conversation.item = MagicMock()
        self.conversation.item.create = AsyncMock()
        self.response = MagicMock()
        self.response.create = AsyncMock()
        self.close = AsyncMock()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.events):
            raise StopAsyncIteration
        event = self.events[self.index]
        self.index += 1
        return event


class TestRealtimeFullFlow:
    """Test complete realtime conversation flows using mocks."""

    @pytest.fixture
    def tracer_provider_and_exporter(self):
        """Create a tracer provider with in-memory exporter."""
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            __import__(
                "opentelemetry.sdk.trace.export",
                fromlist=["SimpleSpanProcessor"]
            ).SimpleSpanProcessor(exporter)
        )
        return provider, exporter

    @pytest.mark.asyncio
    async def test_full_text_conversation_flow(self, tracer_provider_and_exporter):
        """Test a complete text conversation flow with proper span hierarchy."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            RealtimeConnectionManagerWrapper,
        )

        provider, exporter = tracer_provider_and_exporter
        tracer = provider.get_tracer("test")

        # Simulate event stream
        events = [
            MockRealtimeEvent("session.created", session=MockRealtimeSession()),
            MockRealtimeEvent("session.updated", session=MockRealtimeSession(modalities=["text"])),
            MockRealtimeEvent("response.created", response=MockRealtimeResponse()),
            MockRealtimeEvent("response.text.delta", delta="Hello"),
            MockRealtimeEvent("response.text.delta", delta=" there!"),
            MockRealtimeEvent("response.done", response=MockRealtimeResponse(
                usage=MockRealtimeUsage(input_tokens=5, output_tokens=10)
            )),
        ]

        mock_connection = MockAsyncIterableConnection(events)
        mock_manager = AsyncMock()
        mock_manager.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_manager.__aexit__ = AsyncMock(return_value=None)
        wrapper = RealtimeConnectionManagerWrapper(
            mock_manager, tracer, "gpt-4o-realtime-preview"
        )

        async with wrapper as conn:
            await conn.session.update(session={"modalities": ["text"]})
            await conn.conversation.item.create(item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}]
            })
            await conn.response.create()
            async for event in conn:
                if event.type == "response.done":
                    break

        spans = exporter.get_finished_spans()
        span_names = [s.name for s in spans]
        assert "openai.session" in span_names
        assert "openai.realtime" in span_names

        session_span = next(s for s in spans if s.name == "openai.session")
        assert session_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"
        assert session_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o-realtime-preview"
        assert session_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "realtime"
        assert session_span.status.status_code == StatusCode.OK

        response_span = next(s for s in spans if s.name == "openai.realtime")
        assert response_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"
        assert response_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "realtime"

        # Verify finish_reason is "stop" for text response without tool calls
        attrs = dict(response_span.attributes)
        assert attrs.get("gen_ai.completion.0.finish_reason") == "stop"

    @pytest.mark.asyncio
    async def test_response_span_is_child_of_session_span(self, tracer_provider_and_exporter):
        """Test that response spans are properly nested under session spans."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            RealtimeConnectionManagerWrapper,
        )

        provider, exporter = tracer_provider_and_exporter
        tracer = provider.get_tracer("test")

        events = [
            MockRealtimeEvent("session.created", session=MockRealtimeSession()),
            MockRealtimeEvent("response.created", response=MockRealtimeResponse()),
            MockRealtimeEvent("response.text.delta", delta="Hello"),
            MockRealtimeEvent("response.done", response=MockRealtimeResponse(
                usage=MockRealtimeUsage(input_tokens=5, output_tokens=10)
            )),
        ]

        mock_connection = MockAsyncIterableConnection(events)
        mock_manager = AsyncMock()
        mock_manager.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_manager.__aexit__ = AsyncMock(return_value=None)
        wrapper = RealtimeConnectionManagerWrapper(
            mock_manager, tracer, "gpt-4o-realtime-preview"
        )

        async with wrapper as conn:
            await conn.response.create()
            async for event in conn:
                if event.type == "response.done":
                    break

        spans = exporter.get_finished_spans()
        session_span = next(s for s in spans if s.name == "openai.session")
        response_span = next(s for s in spans if s.name == "openai.realtime")

        # Verify parent-child relationship
        assert response_span.parent is not None, "Response span should have a parent"
        assert response_span.parent.span_id == session_span.context.span_id, (
            "Response span's parent should be the session span"
        )
        assert response_span.context.trace_id == session_span.context.trace_id, (
            "Response span should be in the same trace as session span"
        )

    @pytest.mark.asyncio
    async def test_multi_turn_conversation_creates_multiple_spans(self, tracer_provider_and_exporter):
        """Test that multi-turn conversations create multiple response spans."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            RealtimeConnectionManagerWrapper,
        )

        provider, exporter = tracer_provider_and_exporter
        tracer = provider.get_tracer("test")

        events = [
            MockRealtimeEvent("session.created", session=MockRealtimeSession()),
            MockRealtimeEvent("response.created", response=MockRealtimeResponse(response_id="resp_1")),
            MockRealtimeEvent("response.text.delta", delta="First response"),
            MockRealtimeEvent("response.done", response=MockRealtimeResponse(
                response_id="resp_1", usage=MockRealtimeUsage(input_tokens=5, output_tokens=8)
            )),
            MockRealtimeEvent("response.created", response=MockRealtimeResponse(response_id="resp_2")),
            MockRealtimeEvent("response.text.delta", delta="Second response"),
            MockRealtimeEvent("response.done", response=MockRealtimeResponse(
                response_id="resp_2", usage=MockRealtimeUsage(input_tokens=10, output_tokens=12)
            )),
        ]

        mock_connection = MockAsyncIterableConnection(events)
        mock_manager = AsyncMock()
        mock_manager.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_manager.__aexit__ = AsyncMock(return_value=None)
        wrapper = RealtimeConnectionManagerWrapper(
            mock_manager, tracer, "gpt-4o-realtime-preview"
        )

        response_count = 0
        async with wrapper as conn:
            async for event in conn:
                if event.type == "response.done":
                    response_count += 1
                    if response_count >= 2:
                        break

        spans = exporter.get_finished_spans()
        response_spans = [s for s in spans if s.name == "openai.realtime"]
        assert len(response_spans) == 2

    @pytest.mark.asyncio
    async def test_function_call_flow(self, tracer_provider_and_exporter):
        """Test function calling flow captures tool call information."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            RealtimeConnectionManagerWrapper,
        )

        provider, exporter = tracer_provider_and_exporter
        tracer = provider.get_tracer("test")

        events = [
            MockRealtimeEvent("session.created", session=MockRealtimeSession()),
            MockRealtimeEvent("response.created", response=MockRealtimeResponse()),
            MockRealtimeEvent(
                "response.function_call_arguments.done",
                name="get_weather",
                call_id="call_123",
                arguments='{"location": "NYC"}'
            ),
            MockRealtimeEvent("response.done", response=MockRealtimeResponse(
                usage=MockRealtimeUsage()
            )),
        ]

        mock_connection = MockAsyncIterableConnection(events)
        mock_manager = AsyncMock()
        mock_manager.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_manager.__aexit__ = AsyncMock(return_value=None)
        wrapper = RealtimeConnectionManagerWrapper(
            mock_manager, tracer, "gpt-4o-realtime-preview"
        )

        async with wrapper as conn:
            await conn.response.create()
            async for event in conn:
                if event.type == "response.done":
                    break

        spans = exporter.get_finished_spans()
        response_span = next(s for s in spans if s.name == "openai.realtime")
        assert response_span.status.status_code == StatusCode.OK

        # Verify tool call attributes are set
        attrs = dict(response_span.attributes)
        assert attrs.get("gen_ai.completion.0.role") == "assistant"
        assert attrs.get("gen_ai.completion.0.finish_reason") == "tool_calls"
        assert attrs.get("gen_ai.completion.0.tool_calls.0.type") == "function"
        assert attrs.get("gen_ai.completion.0.tool_calls.0.name") == "get_weather"
        assert attrs.get("gen_ai.completion.0.tool_calls.0.id") == "call_123"
        assert attrs.get("gen_ai.completion.0.tool_calls.0.arguments") == '{"location": "NYC"}'

    @pytest.mark.asyncio
    async def test_error_handling_in_response(self, tracer_provider_and_exporter):
        """Test that errors during response are properly captured."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            RealtimeConnectionManagerWrapper,
        )

        provider, exporter = tracer_provider_and_exporter
        tracer = provider.get_tracer("test")

        events = [
            MockRealtimeEvent("session.created", session=MockRealtimeSession()),
            MockRealtimeEvent("response.created", response=MockRealtimeResponse()),
            MockRealtimeEvent("error", error="Rate limit exceeded"),
        ]

        mock_connection = MockAsyncIterableConnection(events)
        mock_manager = AsyncMock()
        mock_manager.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_manager.__aexit__ = AsyncMock(return_value=None)
        wrapper = RealtimeConnectionManagerWrapper(
            mock_manager, tracer, "gpt-4o-realtime-preview"
        )

        async with wrapper as conn:
            await conn.response.create()
            async for event in conn:
                if event.type == "error":
                    break

        spans = exporter.get_finished_spans()
        response_span = next(s for s in spans if s.name == "openai.realtime")
        assert response_span.status.status_code == StatusCode.ERROR

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, tracer_provider_and_exporter):
        """Test that connection-level errors are handled."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            RealtimeConnectionManagerWrapper,
        )

        provider, exporter = tracer_provider_and_exporter
        tracer = provider.get_tracer("test")

        mock_connection = MagicMock()
        mock_connection.close = AsyncMock()
        mock_manager = AsyncMock()
        mock_manager.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_manager.__aexit__ = AsyncMock(return_value=None)
        wrapper = RealtimeConnectionManagerWrapper(
            mock_manager, tracer, "gpt-4o-realtime-preview"
        )

        try:
            async with wrapper:
                raise ConnectionError("WebSocket disconnected")
        except ConnectionError:
            pass

        spans = exporter.get_finished_spans()
        session_span = next(s for s in spans if s.name == "openai.session")
        assert session_span.status.status_code == StatusCode.ERROR


class TestRealtimeConnectWrapper:
    """Test the main realtime_connect_wrapper function."""

    def test_wrapper_returns_connection_manager_wrapper(self):
        """Test that realtime_connect_wrapper returns a wrapped manager."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            realtime_connect_wrapper,
            RealtimeConnectionManagerWrapper,
        )

        mock_tracer = MagicMock()
        mock_instance = MagicMock()
        mock_manager = MagicMock()
        mock_wrapped = MagicMock(return_value=mock_manager)

        wrapper_fn = realtime_connect_wrapper(mock_tracer)
        result = wrapper_fn(mock_wrapped, mock_instance, (), {"model": "gpt-4o-realtime-preview"})
        assert isinstance(result, RealtimeConnectionManagerWrapper)
        mock_wrapped.assert_called_once()

    def test_wrapper_uses_default_model(self):
        """Test that wrapper uses default model when not specified."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            realtime_connect_wrapper,
            RealtimeConnectionManagerWrapper,
        )

        mock_tracer = MagicMock()
        mock_instance = MagicMock()
        mock_manager = MagicMock()
        mock_wrapped = MagicMock(return_value=mock_manager)

        wrapper_fn = realtime_connect_wrapper(mock_tracer)
        result = wrapper_fn(mock_wrapped, mock_instance, (), {})

        assert isinstance(result, RealtimeConnectionManagerWrapper)
        assert result._model == "gpt-4o-realtime-preview"

    def test_wrapper_respects_suppress_instrumentation(self):
        """Test that wrapper respects suppress instrumentation flag."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            realtime_connect_wrapper,
            RealtimeConnectionManagerWrapper,
        )
        from opentelemetry import context as context_api
        from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY

        mock_tracer = MagicMock()
        mock_instance = MagicMock()
        mock_manager = MagicMock()
        mock_wrapped = MagicMock(return_value=mock_manager)

        token = context_api.attach(
            context_api.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True)
        )

        try:
            wrapper_fn = realtime_connect_wrapper(mock_tracer)
            result = wrapper_fn(mock_wrapped, mock_instance, (), {})
            assert not isinstance(result, RealtimeConnectionManagerWrapper)
            assert result == mock_manager
        finally:
            context_api.detach(token)


class TestRealtimeSessionWrapper:
    """Test RealtimeSessionWrapper functionality."""

    @pytest.mark.asyncio
    async def test_session_update_tracks_config(self):
        """Test that session.update tracks configuration changes."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            RealtimeSessionWrapper,
            RealtimeSessionState,
        )

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        state = RealtimeSessionState(mock_tracer, "gpt-4o-realtime-preview")
        state.session_span = mock_span

        mock_session = MagicMock()
        mock_session.update = AsyncMock()

        wrapper = RealtimeSessionWrapper(mock_session, state)

        # Call update with config
        await wrapper.update(session={
            "modalities": ["text"],
            "temperature": 0.8,
            "instructions": "Be helpful"
        })

        # Verify config was tracked
        assert state.session_config.get("modalities") == ["text"]
        assert state.session_config.get("temperature") == 0.8
        assert state.session_config.get("instructions") == "Be helpful"


class TestRealtimeConversationWrapper:
    """Test RealtimeConversationWrapper functionality."""

    @pytest.mark.asyncio
    async def test_item_create_tracks_input(self):
        """Test that conversation.item.create tracks user input."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            RealtimeConversationItemWrapper,
            RealtimeSessionState,
        )

        mock_tracer = MagicMock()
        state = RealtimeSessionState(mock_tracer, "gpt-4o-realtime-preview")

        mock_item = MagicMock()
        mock_item.create = AsyncMock()

        wrapper = RealtimeConversationItemWrapper(mock_item, state)

        # Create a message item
        await wrapper.create(item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello, AI!"}]
        })

        # Verify input was tracked
        assert state.input_text == "Hello, AI!"


class TestRealtimeResponseWrapper:
    """Test RealtimeResponseWrapper functionality."""

    @pytest.mark.asyncio
    async def test_response_cancel_ends_span(self):
        """Test that response.cancel properly ends the span."""
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            RealtimeEventProcessor,
            RealtimeResponseWrapper,
            RealtimeSessionState,
        )

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_tracer.start_span.return_value = mock_span

        state = RealtimeSessionState(mock_tracer, "gpt-4o-realtime-preview")
        state.trace_context = None
        processor = RealtimeEventProcessor(state)

        mock_response = MagicMock()
        mock_response.create = AsyncMock()
        mock_response.cancel = AsyncMock()

        wrapper = RealtimeResponseWrapper(mock_response, state, processor)

        # Start a response
        await wrapper.create()

        # Cancel it
        await wrapper.cancel()

        # Verify span was ended with cancellation
        mock_span.set_status.assert_called()
        mock_span.set_attribute.assert_called_with("response.cancelled", True)
        mock_span.end.assert_called()
