"""Unit tests configuration module."""

import os
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.instrumentation.openai_agents import (
    OpenAIAgentsInstrumentor,
)
from opentelemetry.trace import set_tracer_provider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry import metrics

from agents import Agent, function_tool, ModelSettings, WebSearchTool
from pydantic import BaseModel

pytest_plugins = []


@pytest.fixture(scope="session")
def exporter():
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    set_tracer_provider(provider)

    OpenAIAgentsInstrumentor().instrument()

    return exporter


@pytest.fixture(autouse=True)
def environment():
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "api-key"


@pytest.fixture(autouse=True)
def clear_exporter(exporter):
    exporter.clear()


@pytest.fixture(scope="session")
def metrics_test_context():
    resource = Resource.create()
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader], resource=resource)
    metrics.set_meter_provider(provider)
    OpenAIAgentsInstrumentor().instrument(meter_provider=provider)
    return provider, reader


@pytest.fixture(scope="session", autouse=True)
def clear_metrics_test_context(metrics_test_context):
    provider, reader = metrics_test_context
    reader.shutdown()
    provider.shutdown()


@pytest.fixture(scope="session")
def test_agent():
    test_agent = Agent(
        name="testAgent",
        instructions="You are a helpful assistant that answers all questions",
        model="gpt-4.1",
        model_settings=ModelSettings(
            temperature=0.3, max_tokens=1024, top_p=0.2, frequency_penalty=1.3
        ),
    )
    return test_agent


@pytest.fixture(scope="session")
def function_tool_agent():
    @function_tool
    async def get_weather(city: str) -> str:
        """Gets the current weather for a specified city."""
        if city == "London":
            return "It's cloudy with 15°C"
        return "Weather not available."

    return Agent(
        name="WeatherAgent",
        instructions=(
            "You get the weather for a city using the get_weather tool."
        ),
        model="gpt-4.1",
        tools=[get_weather],
    )


@pytest.fixture(scope="session")
def web_search_tool_agent():
    return Agent(
        name="SearchAgent",
        instructions="You search the web for information.",
        model="gpt-4.1",
        tools=[WebSearchTool()],
    )


@pytest.fixture(scope="session")
def handoff_agent():

    agent_a = Agent(name="AgentA", instructions="Agent A does something.",
                    model="gpt-4.1")
    agent_b = Agent(name="AgentB", instructions="Agent B does something else.",
                    model="gpt-4.1")

    class HandoffExample(BaseModel):
        message: str

    handoff_tool_a = agent_a.as_tool(
        tool_name="handoff_to_agent_a",
        tool_description="Handoff to Agent A for specific tasks",
    )
    handoff_tool_b = agent_b.as_tool(
        tool_name="handoff_to_agent_b",
        tool_description="Handoff to Agent B for different tasks",
    )

    triage_agent = Agent(
        name="TriageAgent",
        instructions="You decide which agent to handoff to.",
        model="gpt-4.1",
        handoffs=[agent_a, agent_b],
        tools=[handoff_tool_a, handoff_tool_b]
    )
    return triage_agent


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization", "api-key"]}
