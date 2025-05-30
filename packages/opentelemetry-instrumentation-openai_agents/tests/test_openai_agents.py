import pytest
from unittest.mock import MagicMock
from opentelemetry.instrumentation.openai_agents \
    import OpenAIAgentsInstrumentor
from agents import Agent, Runner
from unittest.mock import AsyncMock, patch
from agents.extensions.models.litellm_model import LitellmModel
from agents import ModelSettings


@pytest.fixture
def mock_instrumentor():
    instrumentor = OpenAIAgentsInstrumentor()
    instrumentor.instrument = MagicMock()
    instrumentor.uninstrument = MagicMock()
    return instrumentor


@pytest.fixture
def test_agent():
    mock_model = LitellmModel(
        model="mock-model",
        api_key="test-key",
    )
    return Agent(
        name="TestAgent",
        instructions="You are a test assistant.",
        model=mock_model,
        model_settings=ModelSettings(
            temperature=0.3, max_tokens=128, top_p=0.8, frequency_penalty=0
        ),
    )


def test_openai_agent(test_agent, mock_instrumentor):
    mock_instrumentor.instrument()
    mock_instrumentor.instrument.assert_called_once()
    print("TEST AGENT:", test_agent)

    assert test_agent.name == "TestAgent"
    assert test_agent.instructions == "You are a test assistant."
    assert test_agent.model.model == "mock-model"
    assert test_agent.model.api_key == "test-key"


@pytest.mark.asyncio
async def test_runner_mocked_output():
    agent = Agent(
        name="MockAgent",
        instructions="Just mock it",
        model="fake-model"
    )
    mock_result = AsyncMock()
    mock_result.final_output = "Hello, this is a mocked response!"

    with patch.object(Runner, "run", return_value=mock_result):
        result = await Runner.run(starting_agent=agent, input="Mock input")
        assert result.final_output == "Hello, this is a mocked response!"
