import io
import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_audio_transcription(instrument_legacy, span_exporter, openai_client):
    # Create a mock audio file (in real test, use VCR cassette)
    audio_file = io.BytesIO(b"fake audio content")
    audio_file.name = "test_audio.mp3"

    openai_client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "openai.audio.transcriptions"

    transcription_span = spans[0]
    assert (
        transcription_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "whisper-1"
    )
    assert (
        transcription_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )


@pytest.mark.vcr
async def test_audio_transcription_async(
    instrument_legacy, span_exporter, async_openai_client
):
    # Create a mock audio file
    audio_file = io.BytesIO(b"fake audio content")
    audio_file.name = "test_audio.mp3"

    await async_openai_client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "openai.audio.transcriptions"

    transcription_span = spans[0]
    assert (
        transcription_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "whisper-1"
    )
    assert (
        transcription_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )
