import base64
import json
from pathlib import Path

import pytest
from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic, AsyncAnthropic
from opentelemetry.semconv_ai import SpanAttributes, Meters


@pytest.mark.vcr
def test_anthropic_completion(exporter, reader):
    client = Anthropic()
    client.completions.create(
        prompt=f"{HUMAN_PROMPT}\nHello world\n{AI_PROMPT}",
        model="claude-instant-1.2",
        max_tokens_to_sample=2048,
        top_p=0.1,
    )
    try:
        client.completions.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = exporter.get_finished_spans()
    assert all(span.name == "anthropic.completion" for span in spans)

    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.user"]
        == f"{HUMAN_PROMPT}\nHello world\n{AI_PROMPT}"
    )
    assert anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_choice_metric = False
    found_duration_metric = False
    found_exception_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes[SpanAttributes.LLM_TOKEN_TYPE] in [
                            "output",
                            "input",
                        ]
                        assert (
                            data_point.attributes[SpanAttributes.LLM_RESPONSE_MODEL]
                            == "claude-instant-1.2"
                        )
                        assert data_point.sum > 0

                if metric.name == Meters.LLM_GENERATION_CHOICES:
                    found_choice_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.value >= 1
                        assert (
                            data_point.attributes[SpanAttributes.LLM_RESPONSE_MODEL]
                            == "claude-instant-1.2"
                        )

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )
                    assert all(
                        data_point.attributes.get(SpanAttributes.LLM_RESPONSE_MODEL)
                        == "claude-instant-1.2"
                        or data_point.attributes.get("error.type") == "TypeError"
                        for data_point in metric.data.data_points
                    )

                if metric.name == Meters.LLM_ANTHROPIC_COMPLETION_EXCEPTIONS:
                    found_exception_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.value == 1
                        assert data_point.attributes["error.type"] == "TypeError"

                assert all(
                    data_point.attributes.get("gen_ai.system") == "anthropic"
                    for data_point in metric.data.data_points
                )

    assert found_token_metric is True
    assert found_choice_metric is True
    assert found_duration_metric is True
    assert found_exception_metric is True


@pytest.mark.vcr
def test_anthropic_message_create(exporter, reader):
    client = Anthropic()
    response = client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-opus-20240229",
    )
    try:
        client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = exporter.get_finished_spans()
    assert all(span.name == "anthropic.chat" for span in spans)

    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response.content[0].text
    )
    assert anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 17
    assert (
        anthropic_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
        + anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_choice_metric = False
    found_duration_metric = False
    found_exception_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes[SpanAttributes.LLM_TOKEN_TYPE] in [
                            "output",
                            "input",
                        ]
                        assert (
                            data_point.attributes[SpanAttributes.LLM_RESPONSE_MODEL]
                            == "claude-3-opus-20240229"
                        )
                        assert data_point.sum > 0

                if metric.name == Meters.LLM_GENERATION_CHOICES:
                    found_choice_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.value >= 1
                        assert (
                            data_point.attributes[SpanAttributes.LLM_RESPONSE_MODEL]
                            == "claude-3-opus-20240229"
                        )

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )
                    assert all(
                        data_point.attributes.get(SpanAttributes.LLM_RESPONSE_MODEL)
                        == "claude-3-opus-20240229"
                        or data_point.attributes.get("error.type") == "TypeError"
                        for data_point in metric.data.data_points
                    )

                if metric.name == Meters.LLM_ANTHROPIC_COMPLETION_EXCEPTIONS:
                    found_exception_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.value == 1
                        assert data_point.attributes["error.type"] == "TypeError"

                assert all(
                    data_point.attributes.get("gen_ai.system") == "anthropic"
                    for data_point in metric.data.data_points
                )

    assert found_token_metric is True
    assert found_choice_metric is True
    assert found_duration_metric is True
    assert found_exception_metric is True


@pytest.mark.vcr
def test_anthropic_multi_modal(exporter):
    client = Anthropic()
    response = client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see?",
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64.b64encode(
                                open(
                                    Path(__file__).parent.joinpath("data/logo.jpg"),
                                    "rb",
                                ).read()
                            ).decode("utf-8"),
                        },
                    },
                ],
            },
        ],
        model="claude-3-opus-20240229",
    )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert anthropic_span.attributes[
        f"{SpanAttributes.LLM_PROMPTS}.0.content"
    ] == json.dumps(
        [
            {"type": "text", "text": "What do you see?"},
            {"type": "image_url", "image_url": {"url": "/some/url"}},
        ]
    )
    assert (anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response.content[0].text
    )
    assert anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 1381
    assert (
        anthropic_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
        + anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_async_multi_modal(exporter):
    client = AsyncAnthropic()
    response = await client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see?",
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64.b64encode(
                                open(
                                    Path(__file__).parent.joinpath("data/logo.jpg"),
                                    "rb",
                                ).read()
                            ).decode("utf-8"),
                        },
                    },
                ],
            },
        ],
        model="claude-3-opus-20240229",
    )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert anthropic_span.attributes[
        f"{SpanAttributes.LLM_PROMPTS}.0.content"
    ] == json.dumps(
        [
            {"type": "text", "text": "What do you see?"},
            {"type": "image_url", "image_url": {"url": "/some/url"}},
        ]
    )
    assert (anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response.content[0].text
    )
    assert anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 1311
    assert (
        anthropic_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
        + anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )


@pytest.mark.vcr
def test_anthropic_message_streaming(exporter, reader):
    client = Anthropic()
    response = client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-haiku-20240307",
        stream=True,
    )

    response_content = ""
    for event in response:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            response_content += event.delta.text

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response_content
    )
    assert anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 8
    assert (
        anthropic_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
        + anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_choice_metric = False
    found_duration_metric = False
    # TODO found_exception_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes[SpanAttributes.LLM_TOKEN_TYPE] in [
                            "output",
                            "input",
                        ]
                        assert (
                            data_point.attributes[SpanAttributes.LLM_RESPONSE_MODEL]
                            == "claude-3-haiku-20240307"
                        )
                        assert data_point.sum > 0

                if metric.name == Meters.LLM_GENERATION_CHOICES:
                    found_choice_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.value >= 1
                        assert (
                            data_point.attributes[SpanAttributes.LLM_RESPONSE_MODEL]
                            == "claude-3-haiku-20240307"
                        )

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )
                    assert all(
                        data_point.attributes.get(SpanAttributes.LLM_RESPONSE_MODEL)
                        == "claude-3-haiku-20240307"
                        or data_point.attributes.get("error.type") == "TypeError"
                        for data_point in metric.data.data_points
                    )

                assert all(
                    data_point.attributes.get("gen_ai.system") == "anthropic"
                    for data_point in metric.data.data_points
                )

    assert found_token_metric is True
    assert found_choice_metric is True
    assert found_duration_metric is True


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_message_create(exporter, reader):
    client = AsyncAnthropic()
    response = await client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-opus-20240229",
    )
    try:
        await client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response.content[0].text
    )
    assert anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 17
    assert (
        anthropic_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
        + anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_choice_metric = False
    found_duration_metric = False
    found_exception_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes[SpanAttributes.LLM_TOKEN_TYPE] in [
                            "output",
                            "input",
                        ]
                        assert (
                            data_point.attributes[SpanAttributes.LLM_RESPONSE_MODEL]
                            == "claude-3-opus-20240229"
                        )
                        assert data_point.sum > 0

                if metric.name == Meters.LLM_GENERATION_CHOICES:
                    found_choice_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.value >= 1
                        assert (
                            data_point.attributes[SpanAttributes.LLM_RESPONSE_MODEL]
                            == "claude-3-opus-20240229"
                        )

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )
                    assert all(
                        data_point.attributes.get(SpanAttributes.LLM_RESPONSE_MODEL)
                        == "claude-3-opus-20240229"
                        or data_point.attributes.get("error.type") == "TypeError"
                        for data_point in metric.data.data_points
                    )

                if metric.name == Meters.LLM_ANTHROPIC_COMPLETION_EXCEPTIONS:
                    found_exception_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.value == 1
                        assert data_point.attributes["error.type"] == "TypeError"

                assert all(
                    data_point.attributes.get("gen_ai.system") == "anthropic"
                    for data_point in metric.data.data_points
                )

    assert found_token_metric is True
    assert found_choice_metric is True
    assert found_duration_metric is True
    assert found_exception_metric is True


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_message_streaming(exporter, reader):
    client = AsyncAnthropic()
    response = await client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-haiku-20240307",
        stream=True,
    )
    response_content = ""
    async for event in response:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            response_content += event.delta.text

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response_content
    )
    assert anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 8
    assert (
        anthropic_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
        + anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_choice_metric = False
    found_duration_metric = False
    # TODO found_exception_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes[SpanAttributes.LLM_TOKEN_TYPE] in [
                            "output",
                            "input",
                        ]
                        assert (
                            data_point.attributes[SpanAttributes.LLM_RESPONSE_MODEL]
                            == "claude-3-haiku-20240307"
                        )
                        assert data_point.sum > 0

                if metric.name == Meters.LLM_GENERATION_CHOICES:
                    found_choice_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.value >= 1
                        assert (
                            data_point.attributes[SpanAttributes.LLM_RESPONSE_MODEL]
                            == "claude-3-haiku-20240307"
                        )

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )
                    assert all(
                        data_point.attributes.get(SpanAttributes.LLM_RESPONSE_MODEL)
                        == "claude-3-haiku-20240307"
                        or data_point.attributes.get("error.type") == "TypeError"
                        for data_point in metric.data.data_points
                    )

                assert all(
                    data_point.attributes.get("gen_ai.system") == "anthropic"
                    for data_point in metric.data.data_points
                )

    assert found_token_metric is True
    assert found_choice_metric is True
    assert found_duration_metric is True
