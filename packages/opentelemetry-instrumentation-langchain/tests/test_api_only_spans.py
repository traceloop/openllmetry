"""
Unit tests for TraceloopCallbackHandler with spans that only implement the
OpenTelemetry API ``Span`` interface.

``end_time`` and ``attributes`` are SDK ``ReadableSpan`` properties, not part
of the API. When the global TracerProvider is not the SDK's (for example a
vendor's OpenTelemetry shim), the handler must still end every span; otherwise
spans with children and LLM spans stay open forever.
"""
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.trace import INVALID_SPAN_CONTEXT

from opentelemetry.instrumentation.langchain.callback_handler import (
    TraceloopCallbackHandler,
)


class ApiOnlySpan(trace.Span):
    """A span that implements the API ``Span`` and nothing more."""

    def __init__(self, name):
        self.name = name
        self.ended = 0
        self._attributes = {}

    def end(self, end_time=None):
        self.ended += 1

    def get_span_context(self):
        return INVALID_SPAN_CONTEXT

    def set_attributes(self, attributes):
        self._attributes.update(attributes)

    def set_attribute(self, key, value):
        self._attributes[key] = value

    def add_event(self, name, attributes=None, timestamp=None):
        pass

    def update_name(self, name):
        self.name = name

    def is_recording(self):
        return self.ended == 0

    def set_status(self, status, description=None):
        pass

    def record_exception(self, exception, attributes=None, timestamp=None, escaped=False):
        pass


class ApiOnlyTracer(trace.Tracer):
    def __init__(self):
        self.started = []

    def start_span(self, name, context=None, kind=trace.SpanKind.INTERNAL, *args, **kwargs):
        span = ApiOnlySpan(name)
        self.started.append(span)
        return span

    def start_as_current_span(self, *args, **kwargs):
        raise NotImplementedError


@pytest.fixture(autouse=True)
def restore_otel_context():
    restore_token = context_api.attach(context_api.get_current())
    yield
    context_api.detach(restore_token)


@pytest.fixture
def tracer():
    return ApiOnlyTracer()


@pytest.fixture
def handler(tracer):
    return TraceloopCallbackHandler(
        tracer=tracer,
        duration_histogram=MagicMock(),
        token_histogram=MagicMock(),
    )


def _llm_result():
    usage = {"token_usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}
    return LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="ok"))]],
        llm_output=usage,
    )


def test_spans_with_children_and_llm_spans_are_ended(handler, tracer):
    graph_id, node_id, llm_id = uuid4(), uuid4(), uuid4()
    handler.on_chain_start({}, {}, run_id=graph_id, name="LangGraph")
    handler.on_chain_start({}, {}, run_id=node_id, parent_run_id=graph_id, name="node")
    handler.on_chat_model_start({}, [[]], run_id=llm_id, parent_run_id=node_id, name="llm")

    handler.on_llm_end(_llm_result(), run_id=llm_id, parent_run_id=node_id)
    handler.on_chain_end({}, run_id=node_id, parent_run_id=graph_id)
    handler.on_chain_end({}, run_id=graph_id)

    assert [span.ended for span in tracer.started] == [1, 1, 1]
    assert handler.spans == {}


def test_llm_metrics_fall_back_to_default_vendor(handler):
    llm_id = uuid4()
    handler.on_chat_model_start({}, [[]], run_id=llm_id, name="llm")

    handler.on_llm_end(_llm_result(), run_id=llm_id)

    handler.duration_histogram.record.assert_called_once()
    assert handler.token_histogram.record.call_count == 2
    assert handler.spans == {}


def test_parent_error_ends_open_children(handler, tracer):
    graph_id, node_id = uuid4(), uuid4()
    handler.on_chain_start({}, {}, run_id=graph_id, name="LangGraph")
    handler.on_chain_start({}, {}, run_id=node_id, parent_run_id=graph_id, name="node")

    handler.on_chain_error(ValueError("boom"), run_id=graph_id)

    assert [span.ended for span in tracer.started] == [1, 1]
    assert handler.spans == {}
