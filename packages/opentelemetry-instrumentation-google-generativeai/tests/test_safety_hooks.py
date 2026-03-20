from types import SimpleNamespace

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
from opentelemetry.instrumentation.google_generativeai import (
    _handle_request,
    _apply_completion_safety,
    _apply_prompt_safety,
)
from opentelemetry.instrumentation.google_generativeai.streaming_safety import (
    GoogleGenerativeAIStreamingSafety,
    build_async_streaming_response,
    build_streaming_response,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

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


def test_prompt_safety_masks_positional_prompt_args():
    _, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text="[PII.prompt]",
            overall_action="MASK",
            findings=[
                SafetyFinding(
                    category="PII",
                    severity="MEDIUM",
                    action="MASK",
                    rule_name="PII.prompt",
                    start=0,
                    end=len(context.text),
                )
            ],
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    with tracer.start_as_current_span("gemini.generate_content") as span:
        updated_args, updated_kwargs = _apply_prompt_safety(
            span,
            ("secret",),
            {},
            "gemini.generate_content",
        )

    assert updated_args == ("[PII.prompt]",)
    assert updated_kwargs == {}


def test_prompt_safety_masks_span_prompt_attributes():
    exporter, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text="[PII.prompt]",
            overall_action="MASK",
            findings=[
                SafetyFinding(
                    category="PII",
                    severity="MEDIUM",
                    action="MASK",
                    rule_name="PII.prompt",
                    start=0,
                    end=len(context.text),
                )
            ],
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    with tracer.start_as_current_span("gemini.generate_content") as span:
        updated_args, updated_kwargs = _apply_prompt_safety(
            span,
            ("secret",),
            {},
            "gemini.generate_content",
        )
        _handle_request(span, updated_args, updated_kwargs, "gemini-2.0", None)

    spans = exporter.get_finished_spans()
    assert spans[0].attributes["gen_ai.prompt.0.content"] == '[{"type": "text", "text": "[PII.prompt]"}]'
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"


def test_completion_safety_masks_candidate_parts():
    exporter, tracer = _test_span()
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text="[SECRET.gemini]",
            overall_action="MASK",
            findings=[
                SafetyFinding(
                    category="SECRET",
                    severity="HIGH",
                    action="MASK",
                    rule_name="SECRET.gemini",
                    start=0,
                    end=len(context.text),
                )
            ],
        )
        if context.location == SafetyLocation.COMPLETION and context.text == "secret"
        else None
    )

    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="secret")])
            )
        ]
    )
    with tracer.start_as_current_span("gemini.generate_content") as span:
        _apply_completion_safety(span, response, "gemini.generate_content")

    assert response.candidates[0].content.parts[0].text == "[SECRET.gemini]"
    spans = exporter.get_finished_spans()
    assert len(spans[0].events) == 1


class _FakeStreamSession:
    def process_chunk(self, text):
        return SafetyResult(text="masked", overall_action="allow", findings=[])

    def flush(self):
        return SafetyResult(text="-tail", overall_action="allow", findings=[])


def test_streaming_helper_masks_parts_and_flushes_tail():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())
    item = SimpleNamespace(
        text="secret",
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="secret")])
            )
        ],
    )

    with tracer.start_as_current_span("gemini.generate_content") as span:
        helper = GoogleGenerativeAIStreamingSafety(span, "gemini.generate_content")
        helper.process_item(item)
        helper.flush_pending_item(item)

    assert item.text == "masked-tail"
    assert item.candidates[0].content.parts[0].text == "masked-tail"


def test_streaming_helper_clears_top_level_text_when_holdback_masks_current_parts():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession()
    )
    item = SimpleNamespace(
        text="raw-visible",
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="secret")])
            )
        ],
    )

    with tracer.start_as_current_span("gemini.generate_content") as span:
        helper = GoogleGenerativeAIStreamingSafety(span, "gemini.generate_content")
        helper.process_item(item)

    assert item.text == "masked"
    assert item.candidates[0].content.parts[0].text == "masked"


def test_streaming_helper_flushes_all_pending_parts():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())
    item = SimpleNamespace(
        text="secretsecret",
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text="secret"),
                        SimpleNamespace(text="secret"),
                    ]
                )
            )
        ],
    )

    with tracer.start_as_current_span("gemini.generate_content") as span:
        helper = GoogleGenerativeAIStreamingSafety(span, "gemini.generate_content")
        helper.process_item(item)
        helper.flush_pending_item(item)

    assert item.candidates[0].content.parts[0].text == "masked-tail"
    assert item.candidates[0].content.parts[1].text == "masked-tail"
    assert item.text == "masked-tailmasked-tail"


def test_streaming_helper_ignores_items_without_text_parts():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())
    item = SimpleNamespace(text="raw", candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace()]))])

    with tracer.start_as_current_span("gemini.generate_content") as span:
        helper = GoogleGenerativeAIStreamingSafety(span, "gemini.generate_content")
        assert helper.process_item(item) is item
        helper.flush_pending_item(SimpleNamespace(candidates=[]))

    assert item.text == "raw"


def test_build_streaming_response_flushes_pending_chunk_and_finalizes():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())
    complete = []
    chunks = [
        SimpleNamespace(
            text="raw-a",
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="a")]))],
        ),
        SimpleNamespace(
            text="raw-b",
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="b")]))],
        ),
    ]

    with tracer.start_as_current_span("gemini.generate_content") as span:
        yielded = list(
            build_streaming_response(
                iter(chunks),
                span=span,
                llm_model="gemini-2.0",
                finalize_response=lambda full_text, final_chunk, llm_model: complete.append(
                    (full_text, final_chunk.text, llm_model)
                ),
            )
        )

    assert [item.text for item in yielded] == ["masked", "masked-tail"]
    assert complete == [("maskedmasked-tail", "masked-tail", "gemini-2.0")]


@pytest.mark.asyncio
async def test_build_async_streaming_response_flushes_pending_chunk_and_finalizes():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())
    complete = []

    async def _response():
        yield SimpleNamespace(
            text="raw-a",
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="a")]))],
        )

    with tracer.start_as_current_span("gemini.generate_content") as span:
        yielded = [
            item
            async for item in build_async_streaming_response(
                _response(),
                span=span,
                llm_model="gemini-2.0",
                finalize_response=lambda full_text, final_chunk, llm_model: complete.append(
                    (full_text, final_chunk.text, llm_model)
                ),
            )
        ]

    assert [item.text for item in yielded] == ["masked-tail"]
    assert complete == [("masked-tail", "masked-tail", "gemini-2.0")]
