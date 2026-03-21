from unittest import mock

import pytest

from opentelemetry.instrumentation.fortifyroot import (
    SafetyResult,
    clear_safety_handlers,
    register_completion_safety_stream_factory,
)
from opentelemetry.instrumentation.watsonx.streaming_runtime import (
    build_and_set_stream_response_delegate,
)
from opentelemetry.instrumentation.watsonx.streaming_safety import (
    WatsonxStreamingSafety,
    build_streaming_response,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytestmark = pytest.mark.fr


def setup_function():
    clear_safety_handlers()


def teardown_function():
    clear_safety_handlers()


def _test_span():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    return exporter, tracer


class _FakeStreamSession:
    def process_chunk(self, text):
        return SafetyResult(text="masked", overall_action="allow", findings=[])

    def flush(self):
        return SafetyResult(text="-tail", overall_action="allow", findings=[])


def test_watsonx_streaming_helper_masks_generated_text_and_flushes_tail():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())
    item = {
        "model_id": "watsonx",
        "results": [
            {
                "generated_text": "secret",
                "input_token_count": 1,
                "generated_token_count": 1,
                "stop_reason": "stop",
            }
        ],
    }

    with tracer.start_as_current_span("watsonx.generate_text_stream") as span:
        helper = WatsonxStreamingSafety(span, "watsonx.generate_text_stream")
        helper.process_item(item)
        helper.flush_pending_item(item)

    assert item["results"][0]["generated_text"] == "masked-tail"


def test_watsonx_streaming_response_fail_opens_when_safety_processing_raises():
    finalized = []
    response = iter(
        [
            {
                "model_id": "watsonx",
                "results": [
                    {
                        "generated_text": "a",
                        "input_token_count": 1,
                        "generated_token_count": 1,
                        "stop_reason": "stop",
                    }
                ],
            }
        ]
    )

    with mock.patch.object(
        WatsonxStreamingSafety,
        "process_item",
        side_effect=RuntimeError("boom"),
    ):
        yielded = list(
            build_streaming_response(
                response,
                span=mock.Mock(),
                raw_flag=False,
                finalize_response=lambda state: finalized.append(state.copy()),
                span_name="watsonx.generate_text_stream",
            )
        )

    assert yielded == ["a"]
    assert finalized == [
        {
            "generated_text": "a",
            "model_id": "watsonx",
            "stop_reason": "stop",
            "generated_token_count": 1,
            "input_token_count": 1,
        }
    ]


def test_watsonx_streaming_runtime_uses_span_name_from_span():
    _, tracer = _test_span()

    with tracer.start_as_current_span("watsonx.custom_stream") as span:
        with mock.patch(
            "opentelemetry.instrumentation.watsonx.streaming_runtime.build_streaming_response",
            return_value=iter(()),
        ) as build_streaming_response_mock:
            list(
                build_and_set_stream_response_delegate(
                    span=span,
                    event_logger=None,
                    response=iter(()),
                    raw_flag=False,
                    token_histogram=None,
                    response_counter=None,
                    duration_histogram=None,
                    start_time=None,
                )
            )

    build_streaming_response_mock.assert_called_once()
    assert build_streaming_response_mock.call_args.kwargs["span_name"] == "watsonx.custom_stream"
