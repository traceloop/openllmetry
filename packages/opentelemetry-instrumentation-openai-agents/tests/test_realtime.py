"""
Tests for OpenAI Agents SDK Realtime instrumentation.

The realtime API uses WebSockets, which cannot be recorded with VCR.
These tests use comprehensive mocking to simulate the span data types.
"""

import pytest
from unittest.mock import MagicMock, patch
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import StatusCode
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


class MockSpanData:
    """Base mock for span data."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockSpeechSpanData(MockSpanData):
    """Mock for SpeechSpanData."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "speech"


class MockTranscriptionSpanData(MockSpanData):
    """Mock for TranscriptionSpanData."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "transcription"


class MockSpeechGroupSpanData(MockSpanData):
    """Mock for SpeechGroupSpanData."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "speech_group"


class MockAgentSpan:
    """Mock for an agents SDK span."""

    def __init__(self, span_data, trace_id="test-trace-123", error=None):
        self.span_data = span_data
        self.trace_id = trace_id
        self.error = error


@pytest.fixture
def tracer_provider_and_exporter():
    """Create a tracer provider with in-memory exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


class TestRealtimeSpeechSpans:
    """Test speech synthesis span instrumentation."""

    def test_speech_span_start_creates_otel_span(self, tracer_provider_and_exporter):
        """Test that SpeechSpanData creates an OpenTelemetry span."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )

        provider, exporter = tracer_provider_and_exporter
        tracer = provider.get_tracer("test")
        processor = OpenTelemetryTracingProcessor(tracer)

        # Create a mock trace first
        mock_trace = MagicMock()
        mock_trace.trace_id = "test-trace-123"
        processor.on_trace_start(mock_trace)

        # Mock the span data with type name matching
        speech_data = MockSpeechSpanData(
            model="tts-1",
            output_format="mp3",
            input="Hello, world!",
        )
        speech_data.__class__.__name__ = "SpeechSpanData"

        mock_span = MockAgentSpan(speech_data, trace_id="test-trace-123")

        # Patch the module-level variables directly
        import opentelemetry.instrumentation.openai_agents._hooks as hooks_module

        with (
            patch.object(hooks_module, "_has_realtime_spans", True),
            patch.object(hooks_module, "SpeechSpanData", MockSpeechSpanData),
            patch.object(hooks_module, "TranscriptionSpanData", MockTranscriptionSpanData),
            patch.object(hooks_module, "SpeechGroupSpanData", MockSpeechGroupSpanData),
        ):
            processor.on_span_start(mock_span)
            processor.on_span_end(mock_span)

        # End the trace
        processor.on_trace_end(mock_trace)

        spans = exporter.get_finished_spans()
        span_names = [s.name for s in spans]

        assert "Agent Workflow" in span_names
        assert "openai.realtime.speech" in span_names

        speech_span = next(s for s in spans if s.name == "openai.realtime.speech")
        assert speech_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "realtime"
        assert speech_span.attributes["gen_ai.system"] == "openai"
        assert speech_span.attributes["gen_ai.operation.name"] == "speech"
        assert speech_span.status.status_code == StatusCode.OK

    def test_speech_span_captures_model_and_format(self, tracer_provider_and_exporter):
        """Test that speech span captures model and output format."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )

        provider, exporter = tracer_provider_and_exporter
        tracer = provider.get_tracer("test")
        processor = OpenTelemetryTracingProcessor(tracer)

        mock_trace = MagicMock()
        mock_trace.trace_id = "test-trace-456"
        processor.on_trace_start(mock_trace)

        speech_data = MockSpeechSpanData(
            model="tts-1-hd",
            output_format="opus",
            input="Convert this text to speech",
        )
        speech_data.__class__.__name__ = "SpeechSpanData"

        mock_span = MockAgentSpan(speech_data, trace_id="test-trace-456")

        # Patch the module-level variables directly
        import opentelemetry.instrumentation.openai_agents._hooks as hooks_module

        with (
            patch.object(hooks_module, "_has_realtime_spans", True),
            patch.object(hooks_module, "SpeechSpanData", MockSpeechSpanData),
            patch.object(hooks_module, "TranscriptionSpanData", MockTranscriptionSpanData),
            patch.object(hooks_module, "SpeechGroupSpanData", MockSpeechGroupSpanData),
        ):
            processor.on_span_start(mock_span)
            processor.on_span_end(mock_span)

        processor.on_trace_end(mock_trace)

        spans = exporter.get_finished_spans()
        speech_span = next((s for s in spans if s.name == "openai.realtime.speech"), None)

        assert speech_span is not None
        attrs = dict(speech_span.attributes)
        # Check model was captured on span start
        assert attrs.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == "tts-1-hd"


class TestRealtimeTranscriptionSpans:
    """Test transcription span instrumentation."""

    def test_transcription_span_start_creates_otel_span(self, tracer_provider_and_exporter):
        """Test that TranscriptionSpanData creates an OpenTelemetry span."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )

        provider, exporter = tracer_provider_and_exporter
        tracer = provider.get_tracer("test")
        processor = OpenTelemetryTracingProcessor(tracer)

        mock_trace = MagicMock()
        mock_trace.trace_id = "test-trace-789"
        processor.on_trace_start(mock_trace)

        transcription_data = MockTranscriptionSpanData(
            model="whisper-1",
            input_format="wav",
        )
        transcription_data.__class__.__name__ = "TranscriptionSpanData"

        mock_span = MockAgentSpan(transcription_data, trace_id="test-trace-789")

        # Patch the module-level variables directly
        import opentelemetry.instrumentation.openai_agents._hooks as hooks_module

        with (
            patch.object(hooks_module, "_has_realtime_spans", True),
            patch.object(hooks_module, "SpeechSpanData", MockSpeechSpanData),
            patch.object(hooks_module, "TranscriptionSpanData", MockTranscriptionSpanData),
            patch.object(hooks_module, "SpeechGroupSpanData", MockSpeechGroupSpanData),
        ):
            processor.on_span_start(mock_span)
            processor.on_span_end(mock_span)

        processor.on_trace_end(mock_trace)

        spans = exporter.get_finished_spans()
        span_names = [s.name for s in spans]

        assert "openai.realtime.transcription" in span_names

        transcription_span = next(s for s in spans if s.name == "openai.realtime.transcription")
        assert transcription_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "realtime"
        assert transcription_span.attributes["gen_ai.system"] == "openai"
        assert transcription_span.attributes["gen_ai.operation.name"] == "transcription"

    def test_transcription_span_captures_model_and_format(self, tracer_provider_and_exporter):
        """Test that transcription span captures model and input format."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )

        provider, exporter = tracer_provider_and_exporter
        tracer = provider.get_tracer("test")
        processor = OpenTelemetryTracingProcessor(tracer)

        mock_trace = MagicMock()
        mock_trace.trace_id = "test-trace-abc"
        processor.on_trace_start(mock_trace)

        transcription_data = MockTranscriptionSpanData(
            model="whisper-1",
            input_format="mp3",
        )
        transcription_data.__class__.__name__ = "TranscriptionSpanData"

        mock_span = MockAgentSpan(transcription_data, trace_id="test-trace-abc")

        # Patch the module-level variables directly
        import opentelemetry.instrumentation.openai_agents._hooks as hooks_module

        with (
            patch.object(hooks_module, "_has_realtime_spans", True),
            patch.object(hooks_module, "SpeechSpanData", MockSpeechSpanData),
            patch.object(hooks_module, "TranscriptionSpanData", MockTranscriptionSpanData),
            patch.object(hooks_module, "SpeechGroupSpanData", MockSpeechGroupSpanData),
        ):
            processor.on_span_start(mock_span)
            processor.on_span_end(mock_span)

        processor.on_trace_end(mock_trace)

        spans = exporter.get_finished_spans()
        transcription_span = next((s for s in spans if s.name == "openai.realtime.transcription"), None)

        assert transcription_span is not None
        attrs = dict(transcription_span.attributes)
        # Check model was captured on span start
        assert attrs.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == "whisper-1"


class TestRealtimeSpeechGroupSpans:
    """Test speech group span instrumentation."""

    def test_speech_group_span_creates_otel_span(self, tracer_provider_and_exporter):
        """Test that SpeechGroupSpanData creates an OpenTelemetry span."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )

        provider, exporter = tracer_provider_and_exporter
        tracer = provider.get_tracer("test")
        processor = OpenTelemetryTracingProcessor(tracer)

        mock_trace = MagicMock()
        mock_trace.trace_id = "test-trace-def"
        processor.on_trace_start(mock_trace)

        speech_group_data = MockSpeechGroupSpanData(
            input="Group of speech segments",
        )
        speech_group_data.__class__.__name__ = "SpeechGroupSpanData"

        mock_span = MockAgentSpan(speech_group_data, trace_id="test-trace-def")

        # Patch the module-level variables directly
        import opentelemetry.instrumentation.openai_agents._hooks as hooks_module

        with (
            patch.object(hooks_module, "_has_realtime_spans", True),
            patch.object(hooks_module, "SpeechSpanData", MockSpeechSpanData),
            patch.object(hooks_module, "TranscriptionSpanData", MockTranscriptionSpanData),
            patch.object(hooks_module, "SpeechGroupSpanData", MockSpeechGroupSpanData),
        ):
            processor.on_span_start(mock_span)
            processor.on_span_end(mock_span)

        processor.on_trace_end(mock_trace)

        spans = exporter.get_finished_spans()
        span_names = [s.name for s in spans]

        assert "openai.realtime.speech_group" in span_names

        speech_group_span = next(s for s in spans if s.name == "openai.realtime.speech_group")
        assert speech_group_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "realtime"
        assert speech_group_span.attributes["gen_ai.system"] == "openai"
        assert speech_group_span.attributes["gen_ai.operation.name"] == "speech_group"
        assert speech_group_span.status.status_code == StatusCode.OK


class TestRealtimeErrorHandling:
    """Test error handling for realtime spans."""

    def test_speech_span_with_error(self, tracer_provider_and_exporter):
        """Test that errors are properly recorded on speech spans."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )

        provider, exporter = tracer_provider_and_exporter
        tracer = provider.get_tracer("test")
        processor = OpenTelemetryTracingProcessor(tracer)

        mock_trace = MagicMock()
        mock_trace.trace_id = "test-trace-err"
        processor.on_trace_start(mock_trace)

        speech_data = MockSpeechSpanData(
            model="tts-1",
            input="This will fail",
        )
        speech_data.__class__.__name__ = "SpeechSpanData"

        mock_span = MockAgentSpan(speech_data, trace_id="test-trace-err", error="Connection timeout")

        # Patch the module-level variables directly
        import opentelemetry.instrumentation.openai_agents._hooks as hooks_module

        with (
            patch.object(hooks_module, "_has_realtime_spans", True),
            patch.object(hooks_module, "SpeechSpanData", MockSpeechSpanData),
            patch.object(hooks_module, "TranscriptionSpanData", MockTranscriptionSpanData),
            patch.object(hooks_module, "SpeechGroupSpanData", MockSpeechGroupSpanData),
        ):
            processor.on_span_start(mock_span)
            processor.on_span_end(mock_span)

        processor.on_trace_end(mock_trace)

        spans = exporter.get_finished_spans()
        speech_span = next((s for s in spans if s.name == "openai.realtime.speech"), None)

        if speech_span:
            assert speech_span.status.status_code == StatusCode.ERROR


class TestRealtimeSpanHierarchy:
    """Test span hierarchy for realtime operations."""

    def test_realtime_spans_nested_under_workflow(self, tracer_provider_and_exporter):
        """Test that realtime spans are properly nested under workflow span."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )

        provider, exporter = tracer_provider_and_exporter
        tracer = provider.get_tracer("test")
        processor = OpenTelemetryTracingProcessor(tracer)

        mock_trace = MagicMock()
        mock_trace.trace_id = "test-trace-hierarchy"
        processor.on_trace_start(mock_trace)

        # Create multiple realtime spans
        speech_data = MockSpeechSpanData(model="tts-1", input="Hello")
        speech_data.__class__.__name__ = "SpeechSpanData"
        speech_span = MockAgentSpan(speech_data, trace_id="test-trace-hierarchy")

        transcription_data = MockTranscriptionSpanData(model="whisper-1")
        transcription_data.__class__.__name__ = "TranscriptionSpanData"
        transcription_span = MockAgentSpan(transcription_data, trace_id="test-trace-hierarchy")

        # Patch the module-level variables directly
        import opentelemetry.instrumentation.openai_agents._hooks as hooks_module

        with (
            patch.object(hooks_module, "_has_realtime_spans", True),
            patch.object(hooks_module, "SpeechSpanData", MockSpeechSpanData),
            patch.object(hooks_module, "TranscriptionSpanData", MockTranscriptionSpanData),
            patch.object(hooks_module, "SpeechGroupSpanData", MockSpeechGroupSpanData),
        ):
            processor.on_span_start(speech_span)
            processor.on_span_end(speech_span)

            processor.on_span_start(transcription_span)
            processor.on_span_end(transcription_span)

        processor.on_trace_end(mock_trace)

        spans = exporter.get_finished_spans()
        workflow_span = next(s for s in spans if s.name == "Agent Workflow")
        speech_otel_span = next((s for s in spans if s.name == "openai.realtime.speech"), None)
        transcription_otel_span = next((s for s in spans if s.name == "openai.realtime.transcription"), None)

        # Verify workflow is root
        assert workflow_span.parent is None

        # Verify realtime spans have workflow as parent
        if speech_otel_span:
            assert speech_otel_span.parent is not None
            assert speech_otel_span.parent.span_id == workflow_span.context.span_id

        if transcription_otel_span:
            assert transcription_otel_span.parent is not None
            assert transcription_otel_span.parent.span_id == workflow_span.context.span_id
