import json
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from opentelemetry.instrumentation.langchain.callback_handler import (
    TraceloopCallbackHandler,
)
from opentelemetry.instrumentation.langchain.utils import TRACELOOP_TRACE_CONTENT
from opentelemetry.semconv_ai import SpanAttributes


@pytest.fixture(autouse=True)
def enable_content_tracing(monkeypatch):
    monkeypatch.setenv(TRACELOOP_TRACE_CONTENT, "true")


@pytest.fixture
def callback_handler(tracer_provider):
    tracer = tracer_provider.get_tracer("test")
    return TraceloopCallbackHandler(tracer, MagicMock(), MagicMock())


def _start_chain(handler, run_id, inputs=None, parent_run_id=None, name="TestChain"):
    handler.on_chain_start(
        serialized={"id": [name], "name": name},
        inputs=inputs or {},
        run_id=run_id,
        parent_run_id=parent_run_id,
        tags=[],
        metadata={},
    )


def test_chain_start_preserves_non_ascii_in_entity_input(callback_handler, span_exporter):
    run_id = uuid4()
    text = "こんにちは世界"
    _start_chain(callback_handler, run_id, inputs={"query": text})

    attr = callback_handler.spans[run_id].span.attributes.get(
        SpanAttributes.TRACELOOP_ENTITY_INPUT
    )
    assert text in attr
    assert "\\u3053" not in attr
    assert json.loads(attr)["inputs"]["query"] == text


def test_chain_end_preserves_non_ascii_in_entity_output(callback_handler, span_exporter):
    run_id = uuid4()
    text = "Résultat: café naïve"
    _start_chain(callback_handler, run_id)
    callback_handler.on_chain_end(outputs={"result": text}, run_id=run_id)

    spans = span_exporter.get_finished_spans()
    attr = spans[-1].attributes.get(SpanAttributes.TRACELOOP_ENTITY_OUTPUT)
    assert text in attr
    assert "\\u00e9" not in attr
    assert json.loads(attr)["outputs"]["result"] == text


def test_chain_with_cjk_characters(callback_handler, span_exporter):
    run_id = uuid4()
    inp = "请问今天天气怎么样？"
    out = "今天天气晴朗，温度适宜。"
    _start_chain(callback_handler, run_id, inputs={"question": inp})
    callback_handler.on_chain_end(outputs={"answer": out}, run_id=run_id)

    spans = span_exporter.get_finished_spans()
    in_attr = spans[-1].attributes.get(SpanAttributes.TRACELOOP_ENTITY_INPUT)
    out_attr = spans[-1].attributes.get(SpanAttributes.TRACELOOP_ENTITY_OUTPUT)

    assert inp in in_attr and "\\u8bf7" not in in_attr
    assert out in out_attr and "\\u4eca" not in out_attr
    assert json.loads(in_attr)["inputs"]["question"] == inp
    assert json.loads(out_attr)["outputs"]["answer"] == out


def test_tool_start_preserves_non_ascii_in_entity_input(callback_handler, span_exporter):
    run_id, parent_run_id = uuid4(), uuid4()
    text = "Ärger mit Ümlauten"
    _start_chain(callback_handler, parent_run_id)
    callback_handler.on_tool_start(
        serialized={"id": ["TestTool"], "name": "TestTool"},
        input_str=text,
        run_id=run_id,
        parent_run_id=parent_run_id,
        tags=[],
        metadata={},
        inputs={"text": text},
    )

    attr = callback_handler.spans[run_id].span.attributes.get(
        SpanAttributes.TRACELOOP_ENTITY_INPUT
    )
    assert text in attr
    assert "\\u00c4" not in attr
    assert json.loads(attr)["input_str"] == text


def test_tool_end_preserves_non_ascii_in_entity_output(callback_handler, span_exporter):
    run_id, parent_run_id = uuid4(), uuid4()
    text = "Résultat: données récupérées avec succès"
    _start_chain(callback_handler, parent_run_id)
    callback_handler.on_tool_start(
        serialized={"id": ["TestTool"], "name": "TestTool"},
        input_str="input",
        run_id=run_id,
        parent_run_id=parent_run_id,
        tags=[],
        metadata={},
        inputs={},
    )
    callback_handler.on_tool_end(output=text, run_id=run_id)

    tool_spans = [s for s in span_exporter.get_finished_spans() if "TestTool" in s.name]
    attr = tool_spans[0].attributes.get(SpanAttributes.TRACELOOP_ENTITY_OUTPUT)
    assert text in attr
    assert "\\u00e9" not in attr
    assert json.loads(attr)["output"] == text


def test_tool_with_arabic_characters(callback_handler, span_exporter):
    run_id, parent_run_id = uuid4(), uuid4()
    inp = "مرحبا بالعالم"
    out = "تم البحث بنجاح"
    _start_chain(callback_handler, parent_run_id)
    callback_handler.on_tool_start(
        serialized={"id": ["ArabicTool"], "name": "ArabicTool"},
        input_str=inp,
        run_id=run_id,
        parent_run_id=parent_run_id,
        tags=[],
        metadata={},
        inputs={"query": inp},
    )
    callback_handler.on_tool_end(output=out, run_id=run_id)

    tool_spans = [s for s in span_exporter.get_finished_spans() if "ArabicTool" in s.name]
    in_attr = tool_spans[0].attributes.get(SpanAttributes.TRACELOOP_ENTITY_INPUT)
    out_attr = tool_spans[0].attributes.get(SpanAttributes.TRACELOOP_ENTITY_OUTPUT)

    assert inp in in_attr and "\\u0645" not in in_attr
    assert out in out_attr and "\\u062a" not in out_attr
    assert json.loads(in_attr)["input_str"] == inp
    assert json.loads(out_attr)["output"] == out


def test_chain_with_mixed_ascii_and_non_ascii(callback_handler, span_exporter):
    run_id = uuid4()
    inp = "User query: 日本語テスト (Japanese test)"
    out = "Response: 成功 (success) - status: OK"
    _start_chain(callback_handler, run_id, inputs={"text": inp})
    callback_handler.on_chain_end(outputs={"text": out}, run_id=run_id)

    spans = span_exporter.get_finished_spans()
    in_attr = spans[-1].attributes.get(SpanAttributes.TRACELOOP_ENTITY_INPUT)
    out_attr = spans[-1].attributes.get(SpanAttributes.TRACELOOP_ENTITY_OUTPUT)

    assert inp in in_attr
    assert out in out_attr
    assert json.loads(in_attr)["inputs"]["text"] == inp
    assert json.loads(out_attr)["outputs"]["text"] == out
