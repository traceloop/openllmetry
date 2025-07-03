import pytest
from unittest.mock import MagicMock
from opentelemetry.instrumentation.openai_agents import (
    OpenAIAgentsInstrumentor,
)
from agents import Runner
from unittest.mock import AsyncMock, patch
from opentelemetry.semconv_ai import SpanAttributes, Meters


@pytest.fixture
def mock_instrumentor():
    instrumentor = OpenAIAgentsInstrumentor()
    instrumentor.instrument = MagicMock()
    instrumentor.uninstrument = MagicMock()
    return instrumentor


@pytest.mark.asyncio
async def test_async_runner_mocked_output(test_agent):

    mock_result = AsyncMock()
    mock_result.final_output = "Hello, this is a mocked response!"

    with patch.object(Runner, "run", return_value=mock_result):
        result = await Runner.run(
            starting_agent=test_agent,
            input="Mock input",
        )
        assert result.final_output == "Hello, this is a mocked response!"


def test_sync_runner_mocked_output(test_agent):
    mock_result = MagicMock()
    mock_result.final_output = "Hello, this is a mocked response!"

    with patch.object(Runner, "run", return_value=mock_result):
        result = Runner.run_sync(starting_agent=test_agent, input="Mock input")
        assert result.final_output == "Hello, this is a mocked response!"


@pytest.mark.vcr
def test_agent_spans(exporter, test_agent):
    query = "What is AI?"
    Runner.run_sync(
        test_agent,
        query,
    )
    spans = exporter.get_finished_spans()

    span = spans[0]

    assert [span.name for span in spans] == [
        "testAgent.agent",
    ]

    assert span.attributes[SpanAttributes.LLM_SYSTEM] == "openai"
    assert (
        span.attributes[SpanAttributes.LLM_REQUEST_MODEL]
        == "gpt-4.1"
    )
    assert span.attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] == 0.3
    assert span.attributes[SpanAttributes.LLM_REQUEST_MAX_TOKENS] == 1024
    assert span.attributes[SpanAttributes.LLM_REQUEST_TOP_P] == 0.2
    assert span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert (
        span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "What is AI?"
    )
    assert span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 24


@pytest.mark.vcr
def test_generate_metrics(metrics_test_context, test_agent):

    provider, reader = metrics_test_context

    query = "What is AI?"
    Runner.run_sync(
        test_agent,
        query,
    )
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics

    assert len(resource_metrics) > 0

    found_token_metric = False
    found_duration_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:

                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert (
                            (
                                data_point.attributes[
                                    SpanAttributes.LLM_TOKEN_TYPE
                                ]
                                in [
                                    "output",
                                    "input",
                                ]
                            )
                        )
                        assert data_point.count > 0
                        assert data_point.sum > 0

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    assert any(
                        data_point.count > 0
                        for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0
                        for data_point in metric.data.data_points
                    )
                assert (
                    metric.data.data_points[0].attributes[
                        SpanAttributes.LLM_SYSTEM
                    ]
                    == "openai"
                )
                assert (
                    metric.data.data_points[0].attributes[
                        SpanAttributes.LLM_RESPONSE_MODEL
                    ]
                    == "gpt-4.1"
                )

        assert found_token_metric is True
        assert found_duration_metric is True
