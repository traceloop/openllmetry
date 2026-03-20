import pytest

from opentelemetry.instrumentation.fortifyroot import (
    SafetyFinding,
    SafetyLocation,
    SafetyResult,
    clear_safety_handlers,
    register_completion_safety_handler,
    register_completion_safety_stream_factory,
    register_prompt_safety_handler,
)
from opentelemetry.instrumentation.watsonx.safety import (
    _apply_completion_safety,
    _apply_prompt_safety,
    _resolve_masked_text,
)
from opentelemetry.instrumentation.watsonx.streaming_safety import (
    WatsonxStreamingSafety,
    _update_stream_state,
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


def test_prompt_safety_masks_prompt_arg():
    _, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text="[PII.prompt]",
            overall_action="MASK",
            findings=[SafetyFinding("PII", "HIGH", "MASK", "PII.prompt", 0, len(context.text))],
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    with tracer.start_as_current_span("watsonx.generate") as span:
        updated_args, _ = _apply_prompt_safety(span, ("secret",), {}, "watsonx.generate")

    assert updated_args[0] == "[PII.prompt]"


def test_completion_safety_masks_generated_text():
    exporter, tracer = _test_span()
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text="[SECRET.output]",
            overall_action="MASK",
            findings=[SafetyFinding("SECRET", "HIGH", "MASK", "SECRET.output", 0, len(context.text))],
        )
        if context.location == SafetyLocation.COMPLETION and context.text == "secret"
        else None
    )

    response = {"results": [{"generated_text": "secret"}]}
    with tracer.start_as_current_span("watsonx.generate") as span:
        _apply_completion_safety(span, response, "watsonx.generate")

    assert response["results"][0]["generated_text"] == "[SECRET.output]"
    assert len(exporter.get_finished_spans()[0].events) == 1


def test_prompt_and_completion_cover_list_paths():
    _, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text=f"[MASKED:{context.text}]",
            overall_action="MASK",
            findings=[SafetyFinding("PII", "HIGH", "MASK", "PII.prompt", 0, len(context.text))],
        )
        if context.location == SafetyLocation.PROMPT
        else None
    )
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text=f"[MASKED:{context.text}]",
            overall_action="MASK",
            findings=[SafetyFinding("SECRET", "HIGH", "MASK", "SECRET.output", 0, len(context.text))],
        )
        if context.location == SafetyLocation.COMPLETION
        else None
    )

    with tracer.start_as_current_span("watsonx.generate") as span:
        _, updated_kwargs = _apply_prompt_safety(
            span, (), {"prompt": ["a", "b", 1]}, "watsonx.generate"
        )
        updated_args, _ = _apply_prompt_safety(
            span, (["x", "y"],), {}, "watsonx.generate"
        )
        responses = [
            {"results": [{"generated_text": "one"}, {"generated_text": "one-b"}]},
            {"results": [{"generated_text": "two"}]},
        ]
        _apply_completion_safety(span, responses, "watsonx.generate")

    assert updated_kwargs["prompt"][:2] == ["[MASKED:a]", "[MASKED:b]"]
    assert updated_args[0][:2] == ["[MASKED:x]", "[MASKED:y]"]
    assert responses[0]["results"][0]["generated_text"] == "[MASKED:one]"
    assert responses[0]["results"][1]["generated_text"] == "[MASKED:one-b]"
    assert responses[1]["results"][0]["generated_text"] == "[MASKED:two]"
    assert _resolve_masked_text("same", None) == ("same", False)


class _FakeStreamSession:
    def process_chunk(self, text):
        return SafetyResult(text="masked", overall_action="allow", findings=[])

    def flush(self):
        return SafetyResult(text="-tail", overall_action="allow", findings=[])


def test_streaming_helper_masks_generated_text_and_flushes_tail():
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


def test_streaming_helper_flush_pending_item_is_noop_without_tail():
    class _NoTailSession:
        def process_chunk(self, text):
            return SafetyResult(text="masked", overall_action="allow", findings=[])

        def flush(self):
            return SafetyResult(text="", overall_action="allow", findings=[])

    _, tracer = _test_span()
    register_completion_safety_stream_factory(lambda _: _NoTailSession())
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

    assert item["results"][0]["generated_text"] == "masked"


def test_build_streaming_response_updates_state_and_supports_raw_and_text_modes():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())
    finalized = []
    response = [
        {
            "model_id": "watsonx",
            "results": [
                {
                    "generated_text": "a",
                    "input_token_count": 1,
                    "generated_token_count": 1,
                    "stop_reason": "streaming",
                }
            ],
        },
        {
            "model_id": "watsonx",
            "results": [
                {
                    "generated_text": "b",
                    "input_token_count": 2,
                    "generated_token_count": 3,
                    "stop_reason": "stop",
                }
            ],
        },
    ]

    with tracer.start_as_current_span("watsonx.generate_text_stream") as span:
        yielded_text = list(
            build_streaming_response(
                iter(response),
                span=span,
                raw_flag=False,
                finalize_response=lambda state: finalized.append(state.copy()),
            )
        )

    assert yielded_text == ["masked", "masked-tail"]
    assert finalized == [
        {
            "generated_text": "maskedmasked-tail",
            "model_id": "watsonx",
            "stop_reason": "stop",
            "generated_token_count": 3,
            "input_token_count": 3,
        }
    ]

    state = {
        "generated_text": "",
        "model_id": "",
        "stop_reason": "",
        "generated_token_count": 0,
        "input_token_count": 0,
    }
    _update_stream_state(state, response[0])
    assert state["generated_text"] == "masked"
    assert state["model_id"] == "watsonx"
