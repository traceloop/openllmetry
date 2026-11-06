"""Tests for CrewAI memory operation instrumentation.

These tests verify that Memory.remember, Memory.recall, Memory.forget,
and Memory.reset produce spans with GenAI memory semantic convention attributes.
"""

from unittest.mock import MagicMock, patch

import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import SpanKind, StatusCode

from opentelemetry.instrumentation.crewai.instrumentation import (
    wrap_memory_remember,
    wrap_memory_recall,
    wrap_memory_forget,
    wrap_memory_reset,
    _GEN_AI_OPERATION_NAME,
    _GEN_AI_PROVIDER_NAME,
    _GEN_AI_MEMORY_SCOPE,
    _GEN_AI_MEMORY_TYPE,
    _GEN_AI_MEMORY_CONTENT,
    _GEN_AI_MEMORY_QUERY,
    _GEN_AI_MEMORY_SEARCH_RESULT_COUNT,
    _GEN_AI_MEMORY_UPDATE_STRATEGY,
    _GEN_AI_MEMORY_IMPORTANCE,
    _ERROR_TYPE,
)


class _InMemoryExporter(SpanExporter):
    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def get_finished_spans(self):
        return list(self.spans)


@pytest.fixture()
def tracer_provider():
    return TracerProvider()


@pytest.fixture()
def exporter(tracer_provider):
    exp = _InMemoryExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exp))
    return exp


@pytest.fixture()
def tracer(tracer_provider):
    return tracer_provider.get_tracer("test")


def _get_span(exporter):
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    return span, dict(span.attributes)


class TestMemoryRemember:
    def test_basic_remember(self, tracer, exporter):
        mock_record = MagicMock()
        mock_record.id = "rec-123"
        wrapped = MagicMock(return_value=mock_record)
        instance = MagicMock()
        instance._root = None

        wrapper = wrap_memory_remember(tracer, None)
        result = wrapper(
            wrapped, instance, ("Agent met with John",),
            {"scope": "/agent/1", "importance": 0.8}
        )

        assert result == mock_record
        span, attrs = _get_span(exporter)
        assert span.name == "update_memory crewai"
        assert span.kind == SpanKind.CLIENT
        assert attrs[_GEN_AI_OPERATION_NAME] == "update_memory"
        assert attrs[_GEN_AI_PROVIDER_NAME] == "crewai"
        assert attrs[_GEN_AI_MEMORY_UPDATE_STRATEGY] == "merge"
        assert attrs[_GEN_AI_MEMORY_IMPORTANCE] == 0.8
        assert attrs["gen_ai.memory.id"] == "rec-123"

    def test_remember_infers_scope_from_root(self, tracer, exporter):
        wrapped = MagicMock(return_value=MagicMock(id="r1"))
        instance = MagicMock()
        instance._root = "/user/alice"

        wrapper = wrap_memory_remember(tracer, None)
        wrapper(wrapped, instance, ("data",), {})

        span, attrs = _get_span(exporter)
        assert attrs[_GEN_AI_MEMORY_SCOPE] == "user"

    def test_remember_captures_content(self, tracer, exporter, monkeypatch):
        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
        wrapped = MagicMock(return_value=MagicMock(id="r1"))
        instance = MagicMock()
        instance._root = None

        wrapper = wrap_memory_remember(tracer, None)
        wrapper(wrapped, instance, ("User likes dark mode",), {})

        span, attrs = _get_span(exporter)
        assert attrs[_GEN_AI_MEMORY_CONTENT] == "User likes dark mode"

    def test_remember_no_content_by_default(self, tracer, exporter, monkeypatch):
        monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False)
        wrapped = MagicMock(return_value=MagicMock(id="r1"))
        instance = MagicMock()
        instance._root = None

        wrapper = wrap_memory_remember(tracer, None)
        wrapper(wrapped, instance, ("secret",), {})

        span, attrs = _get_span(exporter)
        assert _GEN_AI_MEMORY_CONTENT not in attrs

    def test_remember_error(self, tracer, exporter):
        wrapped = MagicMock(side_effect=RuntimeError("storage error"))
        instance = MagicMock()
        instance._root = None

        wrapper = wrap_memory_remember(tracer, None)
        with pytest.raises(RuntimeError, match="storage error"):
            wrapper(wrapped, instance, ("data",), {})

        span, attrs = _get_span(exporter)
        assert span.status.status_code == StatusCode.ERROR
        assert attrs[_ERROR_TYPE] == "RuntimeError"


class TestMemoryRecall:
    def test_basic_recall(self, tracer, exporter):
        results = [MagicMock(), MagicMock()]
        wrapped = MagicMock(return_value=results)
        instance = MagicMock()
        instance._root = "/agent/bot-1"

        wrapper = wrap_memory_recall(tracer, None)
        result = wrapper(wrapped, instance, ("onboarding process",), {"limit": 5})

        assert result == results
        span, attrs = _get_span(exporter)
        assert span.name == "search_memory crewai"
        assert attrs[_GEN_AI_OPERATION_NAME] == "search_memory"
        assert attrs[_GEN_AI_PROVIDER_NAME] == "crewai"
        assert attrs[_GEN_AI_MEMORY_SCOPE] == "agent"
        assert attrs[_GEN_AI_MEMORY_SEARCH_RESULT_COUNT] == 2

    def test_recall_captures_query(self, tracer, exporter, monkeypatch):
        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
        wrapped = MagicMock(return_value=[])
        instance = MagicMock()
        instance._root = None

        wrapper = wrap_memory_recall(tracer, None)
        wrapper(wrapped, instance, ("find meeting notes",), {})

        span, attrs = _get_span(exporter)
        assert attrs[_GEN_AI_MEMORY_QUERY] == "find meeting notes"

    def test_recall_error(self, tracer, exporter):
        wrapped = MagicMock(side_effect=TimeoutError("vector db timeout"))
        instance = MagicMock()
        instance._root = None

        wrapper = wrap_memory_recall(tracer, None)
        with pytest.raises(TimeoutError):
            wrapper(wrapped, instance, ("query",), {})

        span, attrs = _get_span(exporter)
        assert attrs[_ERROR_TYPE] == "TimeoutError"


class TestMemoryForget:
    def test_basic_forget(self, tracer, exporter):
        wrapped = MagicMock(return_value=3)
        instance = MagicMock()
        instance._root = "/user/alice"

        wrapper = wrap_memory_forget(tracer, None)
        result = wrapper(wrapped, instance, (), {"scope": "/user/alice/temp"})

        assert result == 3
        span, attrs = _get_span(exporter)
        assert span.name == "delete_memory crewai"
        assert attrs[_GEN_AI_OPERATION_NAME] == "delete_memory"
        assert attrs[_GEN_AI_MEMORY_SCOPE] == "user"
        assert attrs["crewai.memory.deleted_count"] == 3

    def test_forget_single_record(self, tracer, exporter):
        wrapped = MagicMock(return_value=1)
        instance = MagicMock()
        instance._root = None

        wrapper = wrap_memory_forget(tracer, None)
        wrapper(wrapped, instance, (), {"record_ids": ["rec-42"]})

        span, attrs = _get_span(exporter)
        assert attrs["gen_ai.memory.id"] == "rec-42"


class TestMemoryReset:
    def test_basic_reset(self, tracer, exporter):
        wrapped = MagicMock(return_value=None)
        instance = MagicMock()
        instance._root = "/agent/bot"

        wrapper = wrap_memory_reset(tracer, None)
        wrapper(wrapped, instance, (), {"scope": "/agent/bot"})

        span, attrs = _get_span(exporter)
        assert span.name == "delete_memory crewai"
        assert attrs[_GEN_AI_OPERATION_NAME] == "delete_memory"
        assert attrs[_GEN_AI_MEMORY_SCOPE] == "agent"
        assert attrs["crewai.memory.reset"] is True
