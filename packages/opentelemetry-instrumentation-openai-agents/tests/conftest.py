"""Unit tests configuration module."""

import os
import sys
import types
import pytest
from unittest.mock import MagicMock
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
from typing import List, Dict, Union

pytest_plugins = []

# Mock traceloop modules before any imports using proper ModuleType
SET_AGENT_NAME_MOCK = MagicMock()

# Create proper module mocks using types.ModuleType for better type safety
mock_traceloop = types.ModuleType('traceloop')
mock_sdk = types.ModuleType('traceloop.sdk')
mock_tracing = types.ModuleType('traceloop.sdk.tracing')

# Set up the module hierarchy and add our mock function
mock_tracing.set_agent_name = SET_AGENT_NAME_MOCK
mock_sdk.tracing = mock_tracing
mock_traceloop.sdk = mock_sdk

# Install mocks in sys.modules before any imports occur
sys.modules['traceloop'] = mock_traceloop
sys.modules['traceloop.sdk'] = mock_sdk
sys.modules['traceloop.sdk.tracing'] = mock_tracing


@pytest.fixture
def mock_set_agent_name():
    """Provide access to the mocked set_agent_name function for test assertions."""
    SET_AGENT_NAME_MOCK.reset_mock()  # Reset mock between tests
    return SET_AGENT_NAME_MOCK


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
    # Disable OpenAI Agents SDK built-in tracing to prevent API calls
    # os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"


@pytest.fixture(autouse=True)
def clear_exporter(exporter):
    exporter.clear()
    # Hook-based approach: cleanup handled automatically


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
        instructions=("You get the weather for a city using the get_weather tool."),
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

    agent_a = Agent(
        name="AgentA", instructions="Agent A does something.", model="gpt-4.1"
    )
    agent_b = Agent(
        name="AgentB", instructions="Agent B does something else.", model="gpt-4.1"
    )

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
        tools=[handoff_tool_a, handoff_tool_b],
    )
    return triage_agent


@pytest.fixture(scope="session")
def recipe_workflow_agents():
    """Create Main Chat Agent and Recipe Editor Agent with function tools for recipe management."""

    class Recipe(BaseModel):
        id: str
        name: str
        ingredients: List[str]
        instructions: List[str]
        prep_time: str
        cook_time: str
        servings: int

    class SearchResponse(BaseModel):
        status: str
        message: str
        recipes: Union[Dict[str, Recipe], None] = None
        recipe_count: Union[int, None] = None
        query: Union[str, None] = None

    class EditResponse(BaseModel):
        status: str
        message: str
        modified_recipe: Union[Recipe, None] = None
        changes_made: Union[List[str], None] = None
        original_recipe: Union[Recipe, None] = None

    # Mock recipe database
    MOCK_RECIPES = {
        "spaghetti_carbonara": {
            "id": "spaghetti_carbonara",
            "name": "Spaghetti Carbonara",
            "ingredients": [
                "400g spaghetti",
                "200g pancetta",
                "4 large eggs",
                "100g Pecorino Romano cheese",
            ],
            "instructions": [
                "Cook spaghetti",
                "Dice pancetta",
                "Whisk eggs with cheese",
            ],
            "prep_time": "10 minutes",
            "cook_time": "15 minutes",
            "servings": 4,
        }
    }

    @function_tool
    async def search_recipes(query: str = "") -> SearchResponse:
        """Search and browse recipes in the database."""
        if "carbonara" in query.lower():
            recipe_data = MOCK_RECIPES["spaghetti_carbonara"]
            recipes_dict = {"spaghetti_carbonara": Recipe(**recipe_data)}
            return SearchResponse(
                status="success",
                message=f'Found 1 recipes matching "{query}"',
                recipes=recipes_dict,
                recipe_count=1,
                query=query,
            )
        return SearchResponse(
            status="success",
            message="No recipes found",
            recipes={},
            recipe_count=0,
            query=query,
        )

    @function_tool
    async def plan_and_apply_recipe_modifications(
        recipe: Recipe, modification_request: str
    ) -> EditResponse:
        """Plan modifications to a recipe based on user request and apply them."""

        if (
            "vegetarian" in modification_request.lower()
            and "carbonara" in recipe.name.lower()
        ):
            modified_recipe = Recipe(
                id=recipe.id,
                name="Vegetarian Carbonara",
                ingredients=[
                    "400g spaghetti",
                    "200g mushrooms",
                    "4 large eggs",
                    "100g Pecorino Romano cheese",
                ],
                instructions=[
                    "Cook spaghetti",
                    "Sauté mushrooms",
                    "Whisk eggs with cheese",
                ],
                prep_time=recipe.prep_time,
                cook_time=recipe.cook_time,
                servings=recipe.servings,
            )
            return EditResponse(
                status="success",
                message="Successfully modified Spaghetti Carbonara to be vegetarian",
                modified_recipe=modified_recipe,
                changes_made=["Replaced pancetta with mushrooms"],
                original_recipe=recipe,
            )

        return EditResponse(status="error", message="Could not modify recipe")

    recipe_editor_agent = Agent(
        name="Recipe Editor Agent",
        instructions="You are a recipe editor specialist. Help users search and modify recipes using your tools.",
        model="gpt-4o",
        tools=[search_recipes, plan_and_apply_recipe_modifications],
    )

    main_chat_agent = Agent(
        name="Main Chat Agent",
        instructions="You handle general conversation and route recipe tasks to the recipe editor agent.",
        model="gpt-4o",
        handoffs=[recipe_editor_agent],
    )

    return main_chat_agent, recipe_editor_agent


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization", "api-key"]}
