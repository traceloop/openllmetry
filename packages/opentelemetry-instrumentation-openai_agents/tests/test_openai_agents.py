import pytest
from unittest.mock import MagicMock
from opentelemetry.instrumentation.openai_agents import (
    OpenAIAgentsInstrumentor,
)
from agents import Agent, Runner
from unittest.mock import AsyncMock, patch
from agents.extensions.models.litellm_model import LitellmModel
from agents import ModelSettings
from opentelemetry.semconv_ai import SpanAttributes


@pytest.fixture
def mock_instrumentor():
    instrumentor = OpenAIAgentsInstrumentor()
    instrumentor.instrument = MagicMock()
    instrumentor.uninstrument = MagicMock()
    return instrumentor


@pytest.mark.asyncio
async def test_async_runner_mocked_output():
    test_agent = Agent(
        name="GroqAgent",
        instructions="You are a helpful assistant that answers all questions",
        model=LitellmModel(
            model="groq/llama3-70b-8192",
        ),
        model_settings=ModelSettings(
            temperature=0.3, max_tokens=1024, top_p=0.2, frequency_penalty=1.3
        ),
    )
    mock_result = AsyncMock()
    mock_result.final_output = "Hello, this is a mocked response!"

    with patch.object(Runner, "run", return_value=mock_result):
        result = await Runner.run(
            starting_agent=test_agent,
            input="Mock input",
        )
        assert result.final_output == "Hello, this is a mocked response!"


def test_sync_runner_mocked_output():
    test_agent = Agent(
        name="GroqAgent",
        instructions="You are a helpful assistant that answers all questions",
        model=LitellmModel(
            model="groq/llama3-70b-8192",
        ),
        model_settings=ModelSettings(
            temperature=0.3, max_tokens=1024, top_p=0.2, frequency_penalty=1.3
        ),
    )
    mock_result = MagicMock()
    mock_result.final_output = "Hello, this is a mocked response!"

    with patch.object(Runner, "run", return_value=mock_result):
        result = Runner.run_sync(starting_agent=test_agent, input="Mock input")
        assert result.final_output == "Hello, this is a mocked response!"


@pytest.mark.vcr
def test_groq_agent_spans(exporter):
    test_agent = Agent(
        name="GroqAgent",
        instructions="You are a helpful assistant that answers all questions",
        model=LitellmModel(
            model="groq/llama3-70b-8192",
        ),
        model_settings=ModelSettings(
            temperature=0.3, max_tokens=1024, top_p=0.2, frequency_penalty=1.3
        ),
    )
    query = "What is AI?"
    Runner.run_sync(
        test_agent,
        query,
    )
    spans = exporter.get_finished_spans()

    span = spans[0]

    assert [span.name for span in spans] == [
        "openai_agents.GroqAgent",
    ]

    assert span.attributes[SpanAttributes.LLM_SYSTEM] == "openai"
    assert (
        span.attributes[SpanAttributes.LLM_REQUEST_MODEL]
        == "groq/llama3-70b-8192"
    )
    assert span.attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] == 0.3
    assert span.attributes[SpanAttributes.LLM_REQUEST_MAX_TOKENS] == 1024
    assert span.attributes[SpanAttributes.LLM_REQUEST_TOP_P] == 0.2
    assert span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert (
        span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "What is AI?"
    )
    assert span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 28
