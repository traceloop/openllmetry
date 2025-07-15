import pytest
import os
from llama_parse import LlamaParse

from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues


@pytest.mark.skipif(
    not pytest.importorskip("llama_parse"),
    reason="llama_parse not installed"
)
@pytest.mark.vcr
def test_llamaparse_load_data_instrumentation(instrument_legacy, span_exporter):
    parser = LlamaParse(api_key=os.environ["LLAMA_CLOUD_API_KEY"], result_type="text")

    parser.load_data("https://arxiv.org/pdf/1706.03762.pdf")

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1

    llamaparse_span = next(
        span for span in spans
        if span.name == "llamaparse.task"
    )

    assert llamaparse_span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == TraceloopSpanKindValues.TASK.value
    assert llamaparse_span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "llamaparse"


@pytest.mark.skipif(
    not pytest.importorskip("llama_parse"),
    reason="llama_parse not installed"
)
@pytest.mark.vcr
@pytest.mark.asyncio
async def test_llamaparse_aload_data_instrumentation(instrument_legacy, span_exporter):
    parser = LlamaParse(api_key=os.environ["LLAMA_CLOUD_API_KEY"], result_type="text")

    await parser.aload_data("https://arxiv.org/pdf/1706.03762.pdf")

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1

    llamaparse_span = next(
        span for span in spans
        if span.name == "llamaparse.task"
    )

    assert llamaparse_span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == TraceloopSpanKindValues.TASK.value
    assert llamaparse_span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "llamaparse"


@pytest.mark.skipif(
    not pytest.importorskip("llama_parse"),
    reason="llama_parse not installed"
)
def test_llamaparse_api_methods_exist():
    parser = LlamaParse(api_key=os.environ["LLAMA_CLOUD_API_KEY"])

    instrumented_methods = [
        'load_data', 'aload_data',
        'get_json_result', 'aget_json',
        'get_images', 'aget_images',
        'get_charts', 'aget_charts'
    ]

    for method_name in instrumented_methods:
        assert hasattr(parser, method_name), f"Method {method_name} does not exist in LlamaParse"
        assert callable(getattr(parser, method_name)), f"Method {method_name} is not callable"


@pytest.mark.skipif(
    not pytest.importorskip("llama_parse"),
    reason="llama_parse not installed"
)
def test_llamaparse_deprecated_methods_dont_exist():
    parser = LlamaParse(api_key=os.environ["LLAMA_CLOUD_API_KEY"])

    non_existent_methods = ['parse', 'aparse']

    for method_name in non_existent_methods:
        assert not hasattr(parser, method_name), f"Method {method_name} should not exist in LlamaParse"
