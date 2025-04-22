import base64
import json
import os
import pytest

from pathlib import Path

from google import genai
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.mark.vcr
def test_gemini_generate_content(exporter: InMemorySpanExporter):
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents="The opposite of hot is ",
    )
    assert "cold" in response.text.lower()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert all(span.name == "gemini.generate_content" for span in spans)

    gemini_span = spans[0]
    assert (
        gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]
        == "user"
    )
    assert (
        gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "The opposite of hot is "
    )
    assert gemini_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == response.text
    assert gemini_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role") == "assistant"

    assert gemini_span.attributes["gen_ai.usage.input_tokens"] == 6
    assert (
        gemini_span.attributes["gen_ai.usage.output_tokens"]
        + gemini_span.attributes["gen_ai.usage.input_tokens"]
        == gemini_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    assert gemini_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gemini-2.0-flash-lite"
    assert gemini_span.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gemini-2.0-flash-lite"


@pytest.mark.vcr
def test_gemini_generate_content_with_image(exporter: InMemorySpanExporter):
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents={
            "role": "user",
            "parts": [
                {
                    "text": "Briefly describe this image"
                },
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(
                            open(
                                Path(__file__).parent.joinpath("data/logo.jpg"),
                                "rb",
                            ).read()
                        ).decode("utf-8")
                    }
                }
            ]
        }
    )
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert all(span.name == "gemini.generate_content" for span in spans)

    gemini_span = spans[0]
    assert (
        gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]
        == "user"
    )

    traced_content = json.loads(gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"])
    assert isinstance(traced_content, list)
    assert len(traced_content) == 2
    assert traced_content[0]["type"] == "text"
    assert traced_content[0]["text"] == "Briefly describe this image"

    # Assert that the image is converted to openai format
    assert traced_content[1]["type"] == "image_url"
    assert traced_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    assert gemini_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == response.text
    assert gemini_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role") == "assistant"

    assert gemini_span.attributes["gen_ai.usage.input_tokens"] == 1295
    assert (
        gemini_span.attributes["gen_ai.usage.output_tokens"]
        + gemini_span.attributes["gen_ai.usage.input_tokens"]
        == gemini_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    assert gemini_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gemini-2.0-flash-lite"
    assert gemini_span.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gemini-2.0-flash-lite"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_gemini_generate_content_with_image_async(exporter: InMemorySpanExporter):
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = await client.aio.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents={
            "role": "user",
            "parts": [
                {
                    "text": "Briefly describe this image"
                },
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(
                            open(
                                Path(__file__).parent.joinpath("data/logo.jpg"),
                                "rb",
                            ).read()
                        ).decode("utf-8")
                    }
                }
            ]
        }
    )
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert all(span.name == "gemini.generate_content" for span in spans)

    gemini_span = spans[0]
    assert (
        gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]
        == "user"
    )

    traced_content = json.loads(gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"])
    assert isinstance(traced_content, list)
    assert len(traced_content) == 2
    assert traced_content[0]["type"] == "text"
    assert traced_content[0]["text"] == "Briefly describe this image"

    # Assert that the image is converted to openai format
    assert traced_content[1]["type"] == "image_url"
    assert traced_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    assert gemini_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == response.text
    assert gemini_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role") == "assistant"

    assert gemini_span.attributes["gen_ai.usage.input_tokens"] == 1295
    assert (
        gemini_span.attributes["gen_ai.usage.output_tokens"]
        + gemini_span.attributes["gen_ai.usage.input_tokens"]
        == gemini_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    assert gemini_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gemini-2.0-flash-lite"
    assert gemini_span.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gemini-2.0-flash-lite"


@pytest.mark.vcr
def test_gemini_generate_content_with_image_and_system_instruction(exporter: InMemorySpanExporter):
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents={
            "role": "user",
            "parts": [
                {
                    "text": "Briefly describe this image"
                },
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(
                            open(
                                Path(__file__).parent.joinpath("data/logo.jpg"),
                                "rb",
                            ).read()
                        ).decode("utf-8")
                    }
                }
            ]
        },
        config={
            "system_instruction": "Be concise and to the point"
        }
    )
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert all(span.name == "gemini.generate_content" for span in spans)

    gemini_span = spans[0]

    assert (
        gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]
        == "system"
    )
    assert (
        gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Be concise and to the point"
    )
    assert (
        gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"]
        == "user"
    )

    traced_content = json.loads(gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"])
    assert isinstance(traced_content, list)
    assert len(traced_content) == 2
    assert traced_content[0]["type"] == "text"
    assert traced_content[0]["text"] == "Briefly describe this image"

    # Assert that the image is converted to openai format
    assert traced_content[1]["type"] == "image_url"
    assert traced_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    assert gemini_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == response.text
    assert gemini_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role") == "assistant"

    assert gemini_span.attributes["gen_ai.usage.input_tokens"] == 1301
    assert (
        gemini_span.attributes["gen_ai.usage.output_tokens"]
        + gemini_span.attributes["gen_ai.usage.input_tokens"]
        == gemini_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    assert gemini_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gemini-2.0-flash-lite"
    assert gemini_span.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gemini-2.0-flash-lite"


@pytest.mark.vcr
def test_gemini_generate_content_system_instruction(exporter: InMemorySpanExporter):
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    system_instruction = "You are a helpful assistant that can answer questions and help with tasks."
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents="The opposite of hot is ",
        config={
            "system_instruction": {
                "text": system_instruction
            }
        }
    )
    assert "cold" in response.text.lower()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert all(span.name == "gemini.generate_content" for span in spans)

    gemini_span = spans[0]
    assert (
        gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]
        == "system"
    )
    assert (
        gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == system_instruction
    )
    assert (
        gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"]
        == "user"
    )
    assert (
        gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"]
        == "The opposite of hot is "
    )
    assert gemini_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == response.text
    assert gemini_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role") == "assistant"

    assert gemini_span.attributes["gen_ai.usage.input_tokens"] == 20
    assert (
        gemini_span.attributes["gen_ai.usage.output_tokens"]
        + gemini_span.attributes["gen_ai.usage.input_tokens"]
        == gemini_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    assert gemini_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gemini-2.0-flash-lite"
    assert gemini_span.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gemini-2.0-flash-lite"


@pytest.mark.vcr
def test_gemini_generate_content_multiple_candidates(exporter: InMemorySpanExporter):
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents="The opposite of hot is ",
        config={
            "candidate_count": 2,
        }
    )
    assert "cold" in response.text.lower()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert all(span.name == "gemini.generate_content" for span in spans)

    gemini_span = spans[0]
    assert (
        gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "The opposite of hot is "
    )
    assert (
        gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]
        == "user"
    )
    assert gemini_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == response.text
    assert gemini_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role") == "assistant"

    second_text = "".join([part.text for part in response.candidates[1].content.parts])
    assert gemini_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.1.content") == second_text
    assert gemini_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.1.role") == "assistant"

    assert gemini_span.attributes["gen_ai.usage.input_tokens"] == 6
    assert (
        gemini_span.attributes["gen_ai.usage.output_tokens"]
        + gemini_span.attributes["gen_ai.usage.input_tokens"]
        == gemini_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    assert gemini_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gemini-2.0-flash-lite"
    assert gemini_span.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gemini-2.0-flash-lite"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_gemini_generate_content_async(exporter: InMemorySpanExporter):
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = await client.aio.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents="The opposite of hot is ",
    )
    assert "cold" in response.text.lower()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert all(span.name == "gemini.generate_content" for span in spans)

    gemini_span = spans[0]
    assert (
        gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "The opposite of hot is "
    )
    assert (
        gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]
        == "user"
    )
    assert gemini_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == response.text
    assert gemini_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role") == "assistant"

    assert gemini_span.attributes["gen_ai.usage.input_tokens"] == 6
    assert (
        gemini_span.attributes["gen_ai.usage.output_tokens"]
        + gemini_span.attributes["gen_ai.usage.input_tokens"]
        == gemini_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    assert gemini_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gemini-2.0-flash-lite"
    assert gemini_span.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gemini-2.0-flash-lite"


@pytest.mark.vcr
def test_gemini_generate_content_streaming(exporter: InMemorySpanExporter):
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = client.models.generate_content_stream(
        model="gemini-2.0-flash-lite",
        contents="Tell me about Google Gemini",
    )
    final_text = ""
    for chunk in response:
        final_text += chunk.text

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert all(span.name == "gemini.generate_content_stream" for span in spans)

    gemini_span = spans[0]
    assert (
        gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me about Google Gemini"
    )
    assert (
        gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]
        == "user"
    )
    assert gemini_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == final_text
    assert gemini_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role") == "assistant"

    assert gemini_span.attributes["gen_ai.usage.input_tokens"] == 6
    assert (
        gemini_span.attributes["gen_ai.usage.output_tokens"]
        + gemini_span.attributes["gen_ai.usage.input_tokens"]
        == gemini_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    assert gemini_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gemini-2.0-flash-lite"
    assert gemini_span.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gemini-2.0-flash-lite"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_gemini_generate_content_streaming_async(exporter: InMemorySpanExporter):
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = await client.aio.models.generate_content_stream(
        model="gemini-2.0-flash-lite",
        contents="Tell me about Google Gemini",
    )
    final_text = ""
    async for chunk in response:
        final_text += chunk.text

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert all(span.name == "gemini.generate_content_stream" for span in spans)

    gemini_span = spans[0]
    assert (
        gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me about Google Gemini"
    )
    assert (
        gemini_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]
        == "user"
    )
    assert gemini_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == final_text
    assert gemini_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role") == "assistant"

    assert gemini_span.attributes["gen_ai.usage.input_tokens"] == 6
    assert (
        gemini_span.attributes["gen_ai.usage.output_tokens"]
        + gemini_span.attributes["gen_ai.usage.input_tokens"]
        == gemini_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    assert gemini_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gemini-2.0-flash-lite"
    assert gemini_span.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gemini-2.0-flash-lite"
