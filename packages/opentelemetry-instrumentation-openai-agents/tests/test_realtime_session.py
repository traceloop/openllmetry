"""Tests for realtime session instrumentation via wrapper patching."""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from opentelemetry.instrumentation.openai_agents._realtime_wrappers import (
    RealtimeTracingState,
    wrap_realtime_session,
    unwrap_realtime_session,
)


@pytest.fixture
def tracer_provider():
    """Create a tracer provider with in-memory exporter."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


@pytest.fixture
def tracer(tracer_provider):
    """Get a tracer from the provider."""
    provider, _ = tracer_provider
    return provider.get_tracer("test-tracer")


class TestRealtimeTracingState:
    """Tests for the RealtimeTracingState class."""

    def test_start_workflow_span(self, tracer, tracer_provider):
        """Test starting a workflow span."""
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)

        span = state.start_workflow_span("Test Agent")

        assert span is not None
        assert state.workflow_span is span

    def test_end_workflow_span(self, tracer, tracer_provider):
        """Test ending a workflow span."""
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)
        state.start_workflow_span("Test Agent")

        state.end_workflow_span()

        assert state.workflow_span is None
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "Realtime Session"

    def test_start_agent_span(self, tracer, tracer_provider):
        """Test starting an agent span."""
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)
        state.start_workflow_span("Test Agent")

        span = state.start_agent_span("Voice Assistant")

        assert span is not None
        assert "Voice Assistant" in state.agent_spans
        assert state.current_agent_name == "Voice Assistant"

    def test_end_agent_span(self, tracer, tracer_provider):
        """Test ending an agent span.

        Note: Agent spans are kept open across turns and only fully ended during cleanup
        to avoid redundant spans for multi-turn conversations.
        """
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)
        state.start_workflow_span("Test Agent")
        state.start_agent_span("Voice Assistant")

        state.end_agent_span("Voice Assistant")
        # Agent span is still active (kept open for potential continuation)
        assert "Voice Assistant" in state.agent_spans

        # Cleanup properly ends all spans
        state.cleanup()
        state.end_workflow_span()

        spans = exporter.get_finished_spans()
        assert any(s.name == "Voice Assistant.agent" for s in spans)

    def test_start_tool_span(self, tracer, tracer_provider):
        """Test starting a tool span."""
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)
        state.start_workflow_span("Test Agent")
        state.start_agent_span("Voice Assistant")

        span = state.start_tool_span("get_weather", "Voice Assistant")

        assert span is not None
        assert "get_weather" in state.tool_spans

    def test_end_tool_span_with_output(self, tracer, tracer_provider):
        """Test ending a tool span with output."""
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)
        state.start_workflow_span("Test Agent")
        state.start_tool_span("get_weather")

        state.end_tool_span("get_weather", output="Sunny, 72F")

        assert "get_weather" not in state.tool_spans
        spans = exporter.get_finished_spans()
        tool_span = next(s for s in spans if s.name == "get_weather.tool")
        assert tool_span.attributes.get("gen_ai.tool.call.result") == "Sunny, 72F"

    def test_create_handoff_span(self, tracer, tracer_provider):
        """Test creating a handoff span."""
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)
        state.start_workflow_span("Test Agent")
        state.start_agent_span("Main Assistant")

        state.create_handoff_span("Main Assistant", "Weather Expert")

        spans = exporter.get_finished_spans()
        handoff_span = next(s for s in spans if "handoff" in s.name)
        assert handoff_span.attributes.get("gen_ai.handoff.from_agent") == "Main Assistant"
        assert handoff_span.attributes.get("gen_ai.handoff.to_agent") == "Weather Expert"

    def test_start_audio_span(self, tracer, tracer_provider):
        """Test starting an audio span."""
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)
        state.start_workflow_span("Test Agent")

        span = state.start_audio_span("item_123", 0)

        assert span is not None
        assert "item_123:0" in state.audio_spans

    def test_end_audio_span(self, tracer, tracer_provider):
        """Test ending an audio span."""
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)
        state.start_workflow_span("Test Agent")
        state.start_audio_span("item_123", 0)

        state.end_audio_span("item_123", 0)

        assert "item_123:0" not in state.audio_spans
        spans = exporter.get_finished_spans()
        assert any(s.name == "openai.realtime" for s in spans)

    def test_record_error_on_agent_span(self, tracer, tracer_provider):
        """Test recording an error on the current agent span."""
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)
        state.start_workflow_span("Test Agent")
        state.start_agent_span("Voice Assistant")

        state.record_error("Connection timeout")

        # The error should be recorded on the agent span using OTel semconv
        assert state.agent_spans["Voice Assistant"].attributes.get("error.message") == "Connection timeout"

    def test_record_prompt(self, tracer, tracer_provider):
        """Test recording a prompt message (buffers for LLM span)."""
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)
        state.start_workflow_span("Test Agent")
        state.start_agent_span("Voice Assistant")

        state.record_prompt("user", "What is the weather?")

        # The prompt should be buffered as (role, content) tuple
        assert ("user", "What is the weather?") in state.pending_prompts

    def test_record_completion_creates_llm_span(self, tracer, tracer_provider):
        """Test that recording a completion creates a dedicated LLM span."""
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)
        state.start_workflow_span("Test Agent")
        state.start_agent_span("Voice Assistant")

        # Record a prompt first, then completion
        state.record_prompt("user", "What is the weather?")
        state.record_completion("assistant", "The weather is sunny.")

        # End spans to export them
        state.end_agent_span("Voice Assistant")
        state.end_workflow_span()

        # Check that an LLM span was created
        spans = exporter.get_finished_spans()
        llm_spans = [s for s in spans if s.name == "openai.realtime"]
        assert len(llm_spans) == 1

        llm_span = llm_spans[0]
        assert llm_span.attributes.get("gen_ai.prompt.0.role") == "user"
        assert llm_span.attributes.get("gen_ai.prompt.0.content") == "What is the weather?"
        assert llm_span.attributes.get("gen_ai.completion.0.role") == "assistant"
        assert llm_span.attributes.get("gen_ai.completion.0.content") == "The weather is sunny."

    def test_multiple_llm_spans(self, tracer, tracer_provider):
        """Test that multiple prompt/completion pairs create multiple LLM spans."""
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)
        state.start_workflow_span("Test Agent")
        state.start_agent_span("Voice Assistant")

        # First exchange
        state.record_prompt("user", "Hello")
        state.record_completion("assistant", "Hi there!")

        # Second exchange
        state.record_prompt("user", "What is the weather?")
        state.record_completion("assistant", "It's sunny.")

        # End spans to export them
        state.end_agent_span("Voice Assistant")
        state.end_workflow_span()

        # Check that two LLM spans were created
        spans = exporter.get_finished_spans()
        llm_spans = [s for s in spans if s.name == "openai.realtime"]
        assert len(llm_spans) == 2

        # First span should have "Hello" and "Hi there!"
        assert llm_spans[0].attributes.get("gen_ai.prompt.0.content") == "Hello"
        assert llm_spans[0].attributes.get("gen_ai.completion.0.content") == "Hi there!"

        # Second span should have "What is the weather?" and "It's sunny."
        assert llm_spans[1].attributes.get("gen_ai.prompt.0.content") == "What is the weather?"
        assert llm_spans[1].attributes.get("gen_ai.completion.0.content") == "It's sunny."

    def test_cleanup_ends_all_spans(self, tracer, tracer_provider):
        """Test that cleanup ends all remaining spans."""
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)
        state.start_workflow_span("Test Agent")
        state.start_agent_span("Voice Assistant")
        state.start_tool_span("get_weather")
        state.start_audio_span("item_123", 0)

        state.cleanup()

        assert len(state.agent_spans) == 0
        assert len(state.tool_spans) == 0
        assert len(state.audio_spans) == 0
        assert state.workflow_span is None

        spans = exporter.get_finished_spans()
        assert len(spans) == 4  # workflow, agent, tool, audio

    def test_record_usage_dict(self, tracer, tracer_provider):
        """Test recording token usage from a dict."""
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)
        state.start_workflow_span("Test Agent")
        state.start_agent_span("Voice Assistant")

        state.record_usage(
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
        )

        assert state.pending_usage is not None
        assert state.pending_usage["input_tokens"] == 100
        assert state.pending_usage["output_tokens"] == 50

    def test_usage_attributes_on_llm_span(self, tracer, tracer_provider):
        """Test that token usage appears on the LLM span."""
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)
        state.start_workflow_span("Test Agent")
        state.start_agent_span("Voice Assistant")

        state.record_prompt("user", "Hello")
        state.record_usage(
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
        )
        state.record_completion("assistant", "Hi there!")

        state.cleanup()
        state.end_workflow_span()

        spans = exporter.get_finished_spans()
        llm_span = next(s for s in spans if s.name == "openai.realtime")

        assert llm_span.attributes.get("gen_ai.usage.input_tokens") == 100
        assert llm_span.attributes.get("gen_ai.usage.output_tokens") == 50

    def test_usage_cleared_after_span(self, tracer, tracer_provider):
        """Test that pending usage is cleared after being applied to a span."""
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)
        state.start_workflow_span("Test Agent")
        state.start_agent_span("Voice Assistant")

        state.record_prompt("user", "Hello")
        state.record_usage({"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})
        state.record_completion("assistant", "Hi!")

        assert state.pending_usage is None

    def test_duplicate_completion_ignored(self, tracer, tracer_provider):
        """Test that duplicate completions are ignored."""
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)
        state.start_workflow_span("Test Agent")
        state.start_agent_span("Voice Assistant")

        state.record_prompt("user", "Hello")
        state.record_completion("assistant", "Hi there!")
        state.record_completion("assistant", "Hi there!")  # Duplicate

        state.cleanup()
        state.end_workflow_span()

        spans = exporter.get_finished_spans()
        llm_spans = [s for s in spans if s.name == "openai.realtime"]
        assert len(llm_spans) == 1


class TestRealtimeSessionWrapping:
    """Tests for the session wrapping functionality."""

    def test_wrap_and_unwrap(self, tracer):
        """Test that wrapping and unwrapping works without errors."""
        # This just tests that the functions don't raise exceptions
        wrap_realtime_session(tracer)
        unwrap_realtime_session()


class TestSpanTiming:
    """Tests for span timing and duration."""

    def test_llm_span_starts_after_agent_span(self, tracer, tracer_provider):
        """Test that LLM span start time is >= agent span start time."""
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)

        state.start_workflow_span("Test Agent")
        state.start_agent_span("Voice Assistant")
        state.record_prompt("user", "Hello")
        state.record_completion("assistant", "Hi!")

        state.cleanup()
        state.end_workflow_span()

        spans = exporter.get_finished_spans()
        agent_span = next(s for s in spans if s.name == "Voice Assistant.agent")
        llm_span = next(s for s in spans if s.name == "openai.realtime")

        assert llm_span.start_time >= agent_span.start_time

    def test_llm_span_ends_before_agent_span(self, tracer, tracer_provider):
        """Test that LLM span ends before or when agent span ends."""
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)

        state.start_workflow_span("Test Agent")
        state.start_agent_span("Voice Assistant")
        state.record_prompt("user", "Hello")
        state.record_completion("assistant", "Hi!")

        state.cleanup()
        state.end_workflow_span()

        spans = exporter.get_finished_spans()
        agent_span = next(s for s in spans if s.name == "Voice Assistant.agent")
        llm_span = next(s for s in spans if s.name == "openai.realtime")

        assert llm_span.end_time <= agent_span.end_time

    def test_tool_span_within_agent_timeframe(self, tracer, tracer_provider):
        """Test that tool span is within agent span timeframe."""
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)

        state.start_workflow_span("Test Agent")
        state.start_agent_span("Voice Assistant")
        state.start_tool_span("get_weather", "Voice Assistant")
        state.end_tool_span("get_weather", output="Sunny")

        state.cleanup()
        state.end_workflow_span()

        spans = exporter.get_finished_spans()
        agent_span = next(s for s in spans if s.name == "Voice Assistant.agent")
        tool_span = next(s for s in spans if s.name == "get_weather.tool")

        assert tool_span.start_time >= agent_span.start_time
        assert tool_span.end_time <= agent_span.end_time

    def test_llm_span_has_duration(self, tracer, tracer_provider):
        """Test that LLM span has non-zero duration."""
        import time

        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)

        state.start_workflow_span("Test Agent")
        state.start_agent_span("Voice Assistant")
        state.record_prompt("user", "Hello")
        time.sleep(0.01)  # Small delay to ensure measurable duration
        state.record_completion("assistant", "Hi!")

        state.cleanup()
        state.end_workflow_span()

        spans = exporter.get_finished_spans()
        llm_span = next(s for s in spans if s.name == "openai.realtime")

        duration_ns = llm_span.end_time - llm_span.start_time
        assert duration_ns > 0


class TestSpanHierarchy:
    """Tests for proper span hierarchy in realtime sessions."""

    def test_tool_span_parented_under_agent(self, tracer, tracer_provider):
        """Test that tool spans are properly parented under agent spans."""
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)

        # Start hierarchy
        state.start_workflow_span("Test Agent")
        state.start_agent_span("Voice Assistant")
        state.start_tool_span("get_weather", "Voice Assistant")

        # End hierarchy
        state.end_tool_span("get_weather")
        state.end_agent_span("Voice Assistant")
        state.cleanup()  # Agent spans are ended during cleanup
        state.end_workflow_span()

        spans = exporter.get_finished_spans()

        # Find spans
        workflow_span = next(s for s in spans if s.name == "Realtime Session")
        agent_span = next(s for s in spans if s.name == "Voice Assistant.agent")
        tool_span = next(s for s in spans if s.name == "get_weather.tool")

        # Verify hierarchy
        assert agent_span.parent.span_id == workflow_span.context.span_id
        assert tool_span.parent.span_id == agent_span.context.span_id

    def test_audio_span_parented_under_current_agent(self, tracer, tracer_provider):
        """Test that audio spans are parented under the current agent."""
        _, exporter = tracer_provider
        state = RealtimeTracingState(tracer)

        state.start_workflow_span("Test Agent")
        state.start_agent_span("Voice Assistant")
        state.start_audio_span("item_123", 0)

        state.end_audio_span("item_123", 0)
        state.end_agent_span("Voice Assistant")
        state.cleanup()  # Agent spans are ended during cleanup
        state.end_workflow_span()

        spans = exporter.get_finished_spans()

        agent_span = next(s for s in spans if s.name == "Voice Assistant.agent")
        audio_span = next(s for s in spans if s.name == "openai.realtime")

        assert audio_span.parent.span_id == agent_span.context.span_id
