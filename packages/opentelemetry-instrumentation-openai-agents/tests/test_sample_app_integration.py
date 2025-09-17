"""Integration test for sample app workflow to verify agent name propagation in complex scenarios."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List
from dataclasses import dataclass
from pydantic import BaseModel

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from agents import Agent, function_tool, Runner, RunContextWrapper


# Sample app models and classes (copied from sample app for testing)
class Recipe(BaseModel):
    id: str
    name: str
    ingredients: List[str]
    instructions: List[str]
    prep_time: str
    cook_time: str
    servings: int


class EditResponse(BaseModel):
    status: str
    message: str
    modified_recipe: Recipe | None = None
    changes_made: List[str] | None = None
    original_recipe: Recipe | None = None


class SearchResponse(BaseModel):
    status: str
    message: str
    recipes: Dict[str, Recipe] | None = None
    recipe_count: int | None = None
    query: str | None = None


class RecipeModificationResult(BaseModel):
    modified_recipe: Recipe
    changes_made: List[str]
    modification_reasoning: str


@dataclass
class ChatContext:
    """Standalone context for the chat application."""
    conversation_history: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []


# Global recipe context for testing
GLOBAL_RECIPE_CONTEXT = {
    "spaghetti_carbonara": {
        "id": "spaghetti_carbonara",
        "name": "Spaghetti Carbonara",
        "ingredients": [
            "400g spaghetti",
            "200g pancetta or guanciale",
            "4 large eggs",
            "100g Pecorino Romano cheese, grated",
            "2 cloves garlic",
            "Black pepper",
            "Salt",
        ],
        "instructions": [
            "Cook spaghetti in salted boiling water until al dente",
            "Dice pancetta and cook in a large pan until crispy",
            "Whisk eggs with grated cheese and black pepper",
            "Drain pasta, reserving 1 cup pasta water",
            "Add hot pasta to the pan with pancetta",
            "Remove from heat, add egg mixture, toss quickly",
            "Add pasta water if needed to create creamy sauce",
            "Serve immediately with extra cheese",
        ],
        "prep_time": "10 minutes",
        "cook_time": "15 minutes",
        "servings": 4,
    }
}


@function_tool
async def search_recipes(
    cw: RunContextWrapper[ChatContext], query: str = ""
) -> SearchResponse:
    """Search and browse recipes in the database."""
    print(f"Searching recipes for query: '{query}'")

    # Return carbonara recipe for testing
    if "carbonara" in query.lower() or not query:
        recipe_data = GLOBAL_RECIPE_CONTEXT["spaghetti_carbonara"]
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
    cw: RunContextWrapper[ChatContext], recipe: Recipe, modification_request: str
) -> EditResponse:
    """Plan modifications to a recipe based on user request and apply them."""
    print(f"Planning and applying modifications for recipe: {recipe.name}")
    print(f"Modification request: {modification_request}")

    # Mock the modification for testing
    if "vegetarian" in modification_request.lower():
        modified_recipe = Recipe(
            id=recipe.id,
            name="Vegetarian Carbonara",
            ingredients=[
                "400g spaghetti",
                "200g mushrooms",
                "4 large eggs",
                "100g Pecorino Romano cheese, grated",
                "2 cloves garlic",
                "Black pepper",
                "Salt",
            ],
            instructions=[
                "Cook spaghetti in salted boiling water until al dente",
                "Dice mushrooms and cook in a large pan until browned",
                "Whisk eggs with grated cheese and black pepper",
                "Drain pasta, reserving 1 cup pasta water",
                "Add hot pasta to the pan with mushrooms",
                "Remove from heat, add egg mixture, toss quickly",
                "Add pasta water if needed to create creamy sauce",
                "Serve immediately with extra cheese",
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


class RecipeEditorAgent(Agent[ChatContext]):
    """Specialized agent for recipe editing and management tasks with function tools."""

    def __init__(self, model: str = "gpt-4o"):
        super().__init__(
            name="Recipe Editor Agent",
            instructions="""
            You are a recipe editor specialist powered by AI. Your role is to:
            1. Help users search and browse recipes using the search_recipes tool
            2. Modify recipes using the plan_and_apply_recipe_modifications tool

            When users want to modify a recipe:
            1. Use search_recipes to find the recipe if they mention it by name
            2. Use plan_and_apply_recipe_modifications to modify the recipe
            """,
            model=model,
            tools=[search_recipes, plan_and_apply_recipe_modifications],
        )


class MainChatAgent(Agent[ChatContext]):
    """Main chat agent that handles general conversation and routes to specialized agents."""

    def __init__(
        self, model: str = "gpt-4o", recipe_editor_agent: RecipeEditorAgent = None
    ):
        super().__init__(
            name="Main Chat Agent",
            instructions="""
            You are a helpful AI assistant that specializes in recipe management and cooking.
            You can handle general cooking conversation and route specialized tasks to expert agents.

            When users ask about recipes, cooking, ingredients, meal planning, or food modifications,
            you will transfer them to the recipe editor agent.

            Keywords that should trigger handoff to recipe editor:
            - "recipe", "recipes", "cooking", "cook"
            - "ingredient", "ingredients", "food"
            - "meal", "dish", "cuisine"
            - "modify", "change", "edit", "update"
            - "vegetarian", "vegan", "gluten-free", "spicy"
            - "search", "find", "browse", "show me"

            For all other conversations, respond helpfully and conversationally.
            """,
            model=model,
            tools=[],
            handoffs=[recipe_editor_agent],
        )


def test_sample_app_agent_name_propagation(exporter):
    """Test that all spans in the sample app workflow have proper agent name propagation."""

    # Clear exporter to start fresh
    exporter.clear()

    # Mock OpenAI API calls to avoid real API calls
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.parsed = RecipeModificationResult(
        modified_recipe=Recipe(
            id="spaghetti_carbonara",
            name="Vegetarian Carbonara",
            ingredients=["400g spaghetti", "200g mushrooms", "4 large eggs"],
            instructions=["Cook spaghetti", "Cook mushrooms", "Mix together"],
            prep_time="10 minutes",
            cook_time="15 minutes",
            servings=4
        ),
        changes_made=["Replaced pancetta with mushrooms"],
        modification_reasoning="Substituted meat with mushrooms for vegetarian option"
    )

    # Set up the agents
    recipe_editor_agent = RecipeEditorAgent()
    main_chat_agent = MainChatAgent(recipe_editor_agent=recipe_editor_agent)

    # Test the workflow: "Can you edit the carbonara recipe to be vegetarian?"
    user_input = "Can you edit the carbonara recipe to be vegetarian?"

    with patch('openai.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_client.beta.chat.completions.parse.return_value = mock_response

        # Run the main agent first
        messages = [{"role": "user", "content": user_input}]

        # This should trigger a handoff to recipe editor agent
        try:
            # Run main agent
            Runner.run_sync(main_chat_agent, messages)

            # Run recipe editor agent (simulating handoff)
            Runner.run_sync(recipe_editor_agent, messages)

        except Exception as e:
            # Some errors are expected due to mocking, but spans should still be created
            print(f"Expected error during test execution: {e}")

    # Get all spans and analyze them
    spans = exporter.get_finished_spans()
    print(f"Total spans captured: {len(spans)}")

    # Print all spans for debugging
    for i, span in enumerate(spans):
        agent_name = span.attributes.get("gen_ai.agent.name", "NO_AGENT_NAME")
        print(f"Span {i+1}: {span.name} - Agent: {agent_name}")
        print(f"  Attributes: {dict(span.attributes)}")
        print()

    # Categorize spans
    workflow_spans = [s for s in spans if s.name == "Agent Workflow"]
    main_agent_spans = [s for s in spans if s.name == "Main Chat Agent.agent"]
    recipe_agent_spans = [s for s in spans if s.name == "Recipe Editor Agent.agent"]
    tool_spans = [s for s in spans if ".tool" in s.name]
    response_spans = [s for s in spans if s.name == "openai.response"]
    handoff_spans = [s for s in spans if "handoff" in s.name.lower()]

    print(f"Workflow spans: {len(workflow_spans)}")
    print(f"Main Chat Agent spans: {len(main_agent_spans)}")
    print(f"Recipe Editor Agent spans: {len(recipe_agent_spans)}")
    print(f"Tool spans: {len(tool_spans)}")
    print(f"Response spans: {len(response_spans)}")
    print(f"Handoff spans: {len(handoff_spans)}")

    # Assertions
    assert len(spans) > 0, "No spans were captured"

    # Workflow spans should NOT have agent name
    for span in workflow_spans:
        assert "gen_ai.agent.name" not in span.attributes, \
            f"Workflow span should not have agent name: {span.name}"

    # Main Chat Agent spans should have correct agent name
    for span in main_agent_spans:
        assert span.attributes.get("gen_ai.agent.name") == "Main Chat Agent", \
            f"Main Chat Agent span missing correct agent name: {span.name}"

    # Recipe Editor Agent spans should have correct agent name
    for span in recipe_agent_spans:
        assert span.attributes.get("gen_ai.agent.name") == "Recipe Editor Agent", \
            f"Recipe Editor Agent span missing correct agent name: {span.name}"

    # All tool spans should have agent name
    for span in tool_spans:
        agent_name = span.attributes.get("gen_ai.agent.name")
        assert agent_name is not None, \
            f"Tool span missing agent name: {span.name}"
        assert agent_name in ["Main Chat Agent", "Recipe Editor Agent"], \
            f"Tool span has incorrect agent name '{agent_name}': {span.name}"

    # All response spans should have agent name
    for span in response_spans:
        agent_name = span.attributes.get("gen_ai.agent.name")
        assert agent_name is not None, \
            f"Response span missing agent name: {span.name}"
        assert agent_name in ["Main Chat Agent", "Recipe Editor Agent"], \
            f"Response span has incorrect agent name '{agent_name}': {span.name}"

    # Handoff spans should have agent name from source agent
    for span in handoff_spans:
        agent_name = span.attributes.get("gen_ai.agent.name")
        if agent_name:  # Handoff spans may or may not have agent name depending on implementation
            assert agent_name in ["Main Chat Agent", "Recipe Editor Agent"], \
                f"Handoff span has incorrect agent name '{agent_name}': {span.name}"

    print("✅ All agent name propagation assertions passed!")


def test_recipe_editor_agent_with_tools(exporter):
    """Test Recipe Editor Agent specifically with tool calls to capture all span types."""

    # Clear exporter to start fresh
    exporter.clear()

    # Set up the recipe editor agent
    recipe_editor_agent = RecipeEditorAgent()

    # Test direct recipe editing workflow
    user_input = "Search for carbonara and make it vegetarian"

    try:
        # Run recipe editor agent directly to trigger tool calls
        messages = [{"role": "user", "content": user_input}]
        Runner.run_sync(recipe_editor_agent, messages)

    except Exception as e:
        # Some errors are expected due to mocking, but spans should still be created
        print(f"Expected error during test execution: {e}")

    # Get all spans and analyze them
    spans = exporter.get_finished_spans()
    print(f"Total spans captured: {len(spans)}")

    # Print all spans for debugging
    for i, span in enumerate(spans):
        agent_name = span.attributes.get("gen_ai.agent.name", "NO_AGENT_NAME")
        print(f"Span {i+1}: {span.name} - Agent: {agent_name}")
        print(f"  Attributes: {dict(span.attributes)}")
        print()

    # Categorize spans
    workflow_spans = [s for s in spans if s.name == "Agent Workflow"]
    recipe_agent_spans = [s for s in spans if s.name == "Recipe Editor Agent.agent"]
    tool_spans = [s for s in spans if ".tool" in s.name]
    response_spans = [s for s in spans if s.name == "openai.response"]

    print(f"Workflow spans: {len(workflow_spans)}")
    print(f"Recipe Editor Agent spans: {len(recipe_agent_spans)}")
    print(f"Tool spans: {len(tool_spans)}")
    print(f"Response spans: {len(response_spans)}")

    # Assertions for this simpler test
    assert len(spans) > 0, "No spans were captured"

    # Workflow spans should NOT have agent name
    for span in workflow_spans:
        assert "gen_ai.agent.name" not in span.attributes, \
            f"Workflow span should not have agent name: {span.name}"

    # Recipe Editor Agent spans should have correct agent name
    for span in recipe_agent_spans:
        assert span.attributes.get("gen_ai.agent.name") == "Recipe Editor Agent", \
            f"Recipe Editor Agent span missing correct agent name: {span.name}"

    # All tool spans should have agent name
    for span in tool_spans:
        agent_name = span.attributes.get("gen_ai.agent.name")
        assert agent_name == "Recipe Editor Agent", \
            f"Tool span has incorrect agent name '{agent_name}': {span.name}"

    # All response spans should have agent name
    for span in response_spans:
        agent_name = span.attributes.get("gen_ai.agent.name")
        assert agent_name == "Recipe Editor Agent", \
            f"Response span has incorrect agent name '{agent_name}': {span.name}"

    print("✅ Recipe Editor Agent tool test passed!")


def test_specific_tool_call_workflow(exporter):
    """Test a very specific workflow that should trigger tool calls."""

    # Clear exporter to start fresh
    exporter.clear()

    # Create a simple agent with just one tool for testing
    from agents import Agent

    @function_tool
    def simple_test_tool(query: str) -> str:
        """A simple test tool that always returns a result."""
        return f"Test result for: {query}"

    class TestAgent(Agent):
        def __init__(self):
            super().__init__(
                name="Test Agent",
                instructions="You are a test agent. Use the simple_test_tool when asked.",
                model="gpt-4o",
                tools=[simple_test_tool]
            )

    test_agent = TestAgent()

    try:
        # Run with a message that should trigger the tool
        messages = [{"role": "user", "content": "Please use the test tool with query 'hello'"}]
        Runner.run_sync(test_agent, messages)

    except Exception as e:
        print(f"Expected error during test execution: {e}")

    # Get all spans and analyze them
    spans = exporter.get_finished_spans()
    print(f"Total spans captured in tool test: {len(spans)}")

    # Print all spans for debugging
    for i, span in enumerate(spans):
        agent_name = span.attributes.get("gen_ai.agent.name", "NO_AGENT_NAME")
        print(f"Span {i+1}: {span.name} - Agent: {agent_name}")

    # Check that we have agent and response spans with correct agent name
    agent_spans = [s for s in spans if s.name == "Test Agent.agent"]
    response_spans = [s for s in spans if s.name == "openai.response"]
    tool_spans = [s for s in spans if "simple_test_tool" in s.name]

    print(f"Agent spans: {len(agent_spans)}")
    print(f"Response spans: {len(response_spans)}")
    print(f"Tool spans: {len(tool_spans)}")

    # Verify agent spans have correct name
    for span in agent_spans:
        assert span.attributes.get("gen_ai.agent.name") == "Test Agent"

    # Verify response spans have agent name
    for span in response_spans:
        agent_name = span.attributes.get("gen_ai.agent.name")
        assert agent_name == "Test Agent", f"Response span missing agent name: {agent_name}"

    # Verify tool spans have agent name (if any were created)
    for span in tool_spans:
        agent_name = span.attributes.get("gen_ai.agent.name")
        assert agent_name == "Test Agent", f"Tool span missing agent name: {agent_name}"

    print("✅ Specific tool call workflow test passed!")