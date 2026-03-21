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
from opentelemetry.instrumentation.vertexai.safety import (
    _apply_completion_safety,
    _apply_prompt_safety,
    _mask_prompt_value,
    _resolve_masked_text,
)
from opentelemetry.instrumentation.vertexai.streaming_safety import (
    VertexAIStreamingSafety,
    build_async_streaming_response,
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


def test_prompt_safety_masks_positional_text_arg():
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

    with tracer.start_as_current_span("vertexai.generate_content") as span:
        updated_args, _ = _apply_prompt_safety(
            span, ("secret",), {}, "vertexai.generate_content"
        )

    assert updated_args[0] == "[PII.prompt]"


def test_completion_safety_masks_response_text():
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

    response = SimpleNamespace(text="secret", candidates=[SimpleNamespace(text="secret")])
    with tracer.start_as_current_span("vertexai.generate_content") as span:
        _apply_completion_safety(span, response, "vertexai.generate_content")

    assert response.text == "secret"
    assert response.candidates[0].text == "[SECRET.output]"
    assert len(exporter.get_finished_spans()[0].events) == 1


def test_prompt_and_completion_cover_contents_parts_and_candidate_parts():
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

    contents = [SimpleNamespace(parts=[SimpleNamespace(text="prompt-part")])]
    response = SimpleNamespace(
        text=None,
        candidates=[
            SimpleNamespace(
                text=None,
                content=SimpleNamespace(parts=[SimpleNamespace(text="completion-part")]),
            )
        ],
    )
    with tracer.start_as_current_span("vertexai.generate_content") as span:
        _, updated_kwargs = _apply_prompt_safety(
            span, (), {"contents": contents}, "vertexai.generate_content"
        )
        _apply_completion_safety(span, response, "vertexai.generate_content")
        assert _mask_prompt_value(
            span, None, span_name="vertexai.generate_content", segment_index=0, segment_role="user"
        ) == (None, False)

    assert updated_kwargs["contents"][0].parts[0].text == "[MASKED:prompt-part]"
    assert response.candidates[0].content.parts[0].text == "[MASKED:completion-part]"
    assert _resolve_masked_text("same", None) == ("same", False)


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
                text="secret",
                content=SimpleNamespace(parts=[SimpleNamespace(text="secret")]),
            )
        ],
    )

    with tracer.start_as_current_span("vertexai.generate_content") as span:
        helper = VertexAIStreamingSafety(span, "vertexai.generate_content")
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

    with tracer.start_as_current_span("vertexai.generate_content") as span:
        helper = VertexAIStreamingSafety(span, "vertexai.generate_content")
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

    with tracer.start_as_current_span("vertexai.generate_content") as span:
        helper = VertexAIStreamingSafety(span, "vertexai.generate_content")
        helper.process_item(item)
        helper.flush_pending_item(item)

    assert item.candidates[0].content.parts[0].text == "masked-tail"
    assert item.candidates[0].content.parts[1].text == "masked-tail"
    assert item.text == "masked-tailmasked-tail"


def test_streaming_helper_ignores_items_without_text_parts():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())
    item = SimpleNamespace(
        text="raw",
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace()]))],
    )

    with tracer.start_as_current_span("vertexai.generate_content") as span:
        helper = VertexAIStreamingSafety(span, "vertexai.generate_content")
        assert helper.process_item(item) is item
        helper.flush_pending_item(SimpleNamespace(candidates=[]))

    assert item.text == "raw"


def test_build_streaming_response_flushes_pending_chunk_and_tracks_usage():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())
    complete = []
    usage = SimpleNamespace(prompt_token_count=1)
    chunks = [
        SimpleNamespace(
            text="raw-a",
            usage_metadata=None,
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="a")]))],
        ),
        SimpleNamespace(
            text="raw-b",
            usage_metadata=usage,
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="b")]))],
        ),
    ]

    with tracer.start_as_current_span("vertexai.generate_content") as span:
        yielded = list(
            build_streaming_response(
                iter(chunks),
                span=span,
                llm_model="vertex-1",
                finalize_response=lambda full_text, token_usage, llm_model: complete.append(
                    (full_text, token_usage, llm_model)
                ),
            )
        )

    assert [item.text for item in yielded] == ["masked", "masked-tail"]
    assert complete == [("maskedmasked-tail", usage, "vertex-1")]


@pytest.mark.asyncio
async def test_build_async_streaming_response_flushes_pending_chunk_and_tracks_usage():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())
    complete = []
    usage = SimpleNamespace(prompt_token_count=1)

    async def _response():
        yield SimpleNamespace(
            text="raw-a",
            usage_metadata=usage,
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="a")]))],
        )

    with tracer.start_as_current_span("vertexai.generate_content") as span:
        yielded = [
            item
            async for item in build_async_streaming_response(
                _response(),
                span=span,
                llm_model="vertex-1",
                finalize_response=lambda full_text, token_usage, llm_model: complete.append(
                    (full_text, token_usage, llm_model)
                ),
            )
        ]

    assert [item.text for item in yielded] == ["masked-tail"]
    assert complete == [("masked-tail", usage, "vertex-1")]
