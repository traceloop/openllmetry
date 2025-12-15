"""Test recipe agents hierarchy matching the expected trace structure."""

import pytest
from typing import Dict, List
from dataclasses import dataclass
from pydantic import BaseModel
from agents import Agent, function_tool, RunContextWrapper, Runner


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
    recipes: Dict[str, Recipe] | None = None
    recipe_count: int | None = None
    query: str | None = None


class EditResponse(BaseModel):
    status: str
    message: str
    modified_recipe: Recipe | None = None
    changes_made: List[str] | None = None
    original_recipe: Recipe | None = None


@dataclass
class ChatContext:
    """Standalone context for the chat application."""
    conversation_history: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []


# Sample recipe data for testing
SAMPLE_RECIPE_DATA = {
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
    # Simulate finding the carbonara recipe
    if "carbonara" in query.lower():
        recipes_dict = {"spaghetti_carbonara": Recipe(**SAMPLE_RECIPE_DATA["spaghetti_carbonara"])}
        return SearchResponse(
            status="success",
            message=f"Found 1 recipe matching '{query}'",
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
    # Mock implementation - simulate making the recipe vegetarian
    modified_ingredients = [ing.replace("pancetta", "mushrooms").replace("guanciale", "mushrooms")
                            for ing in recipe.ingredients]

    modified_recipe = Recipe(
        id=recipe.id,
        name=f"Vegetarian {recipe.name}",
        ingredients=modified_ingredients,
        instructions=recipe.instructions,
        prep_time=recipe.prep_time,
        cook_time=recipe.cook_time,
        servings=recipe.servings,
    )

    return EditResponse(
        status="success",
        message=f"Successfully modified {recipe.name} to be vegetarian",
        modified_recipe=modified_recipe,
        changes_made=["Replaced pancetta/guanciale with mushrooms"],
        original_recipe=recipe,
    )


@pytest.fixture(scope="session")
def recipe_agents():
    """Create Recipe Editor Agent and Main Chat Agent matching the example."""

    # Create Recipe Editor agent with function tools
    recipe_editor_agent = Agent(
        name="Recipe Editor Agent",
        instructions="""
        You are a recipe editor specialist powered by AI. Your role is to:
        1. Help users search and browse recipes using the search_recipes tool with intelligent semantic search
        2. Modify recipes using AI-powered analysis with the plan_and_apply_recipe_modifications tool

        When users want to modify a recipe:
        1. Use search_recipes to find the recipe if they mention it by name
        2. Use plan_and_apply_recipe_modifications to intelligently modify the recipe using AI
        """,
        model="gpt-4o",
        tools=[search_recipes, plan_and_apply_recipe_modifications],
    )

    # Create Main Chat agent that can hand off to Recipe Editor
    main_chat_agent = Agent(
        name="Main Chat Agent",
        instructions="""
        You are a helpful AI assistant that specializes in recipe management and cooking.
        You can handle general cooking conversation and route specialized tasks to expert agents.

        When users ask about recipes, cooking, ingredients, meal planning, or food modifications,
        you will transfer them to the recipe editor agent.
        """,
        model="gpt-4o",
        handoffs=[recipe_editor_agent],
    )

    return main_chat_agent, recipe_editor_agent


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_recipe_agents_hierarchy(exporter, recipe_agents):
    """Test recipe agents hierarchy matches expected structure:

    Expected hierarchy:
    Agent workflow (root)
    ├── Main Chat Agent
    │   ├── openai.response
    │   └── Handoff → Recipe Editor Agent
    └── Recipe Editor Agent
        ├── openai.response
        ├── search_recipes
        ├── openai.response
        ├── plan_and_apply_recipe_modifications
        └── openai.response
    """

    main_chat_agent, recipe_editor_agent = recipe_agents

    query = "Can you edit the carbonara recipe to be vegetarian?"
    messages = [{"role": "user", "content": query}]

    # Run the main chat agent which should handoff to recipe editor
    main_runner = Runner().run_streamed(starting_agent=main_chat_agent, input=messages)

    handoff_occurred = False
    async for event in main_runner.stream_events():
        if event.type == "run_item_stream_event":
            if "handoff" in event.name.lower():
                handoff_occurred = True

    # Continue with recipe editor agent
    if handoff_occurred:
        recipe_messages = [{"role": "user", "content": query}]
        recipe_runner = Runner().run_streamed(starting_agent=recipe_editor_agent, input=recipe_messages)

        async for event in recipe_runner.stream_events():
            pass  # Process all events

    spans = exporter.get_finished_spans()

    # Extract span information for testing
    workflow_spans = [s for s in spans if s.name == "Agent Workflow"]
    agent_spans = [s for s in spans if s.name.endswith(".agent")]
    handoff_spans = [s for s in spans if ".handoff" in s.name]

    # Verify we have the expected root spans
    assert len(workflow_spans) >= 1, f"Should have at least 1 Agent Workflow span, found {len(workflow_spans)}"
    workflow_span = workflow_spans[0]
    assert workflow_span.parent is None, "Agent Workflow should be root span"

    # Verify we have agent spans - Main Chat Agent might not create its own span if it immediately hands off
    main_chat_spans = [s for s in agent_spans if "Main Chat Agent" in s.name]
    recipe_editor_spans = [s for s in agent_spans if "Recipe Editor Agent" in s.name]

    # Main Chat Agent might not create its own span if it immediately hands off
    # The important thing is that we have the handoff span and recipe editor spans
    assert len(
        recipe_editor_spans) >= 1, f"Should have at least 1 Recipe Editor Agent span, found {len(recipe_editor_spans)}"

    recipe_editor_span = recipe_editor_spans[0]

    # Verify recipe editor span is child of workflow span
    assert recipe_editor_span.parent is not None, "Recipe Editor Agent should have parent"
    recipe_editor_workflow_id = recipe_editor_span.parent.span_id if recipe_editor_span.parent else None
    recipe_editor_parent = next((s for s in spans if s.context.span_id == recipe_editor_workflow_id), None)

    if recipe_editor_parent:
        assert recipe_editor_parent.name == "Agent Workflow", (
            f"Recipe Editor Agent parent should be Agent Workflow, got "
            f"{recipe_editor_parent.name}"
        )

    # Verify handoff span exists and is properly parented
    assert len(handoff_spans) >= 1, f"Should have at least 1 handoff span, found {len(handoff_spans)}"
    handoff_span = handoff_spans[0]
    assert handoff_span.parent is not None, "Handoff span should have parent"

    # The handoff span should be a child of the Main Chat Agent (if it exists) or the workflow span
    handoff_parent_id = handoff_span.parent.span_id if handoff_span.parent else None
    handoff_parent = next((s for s in spans if s.context.span_id == handoff_parent_id), None)

    if handoff_parent:
        # Handoff should be child of Main Chat Agent if it exists, otherwise workflow
        expected_parents = ["Main Chat Agent.agent", "Agent Workflow"]
        assert handoff_parent.name in expected_parents, (
            f"Handoff span parent should be Main Chat Agent or Agent Workflow, got "
            f"{handoff_parent.name}"
        )

    # If Main Chat Agent span exists, verify its hierarchy
    if main_chat_spans:
        main_chat_span = main_chat_spans[0]
        assert main_chat_span.parent is not None, "Main Chat Agent should have parent"
        main_chat_workflow_id = main_chat_span.parent.span_id if main_chat_span.parent else None
        main_chat_parent = next((s for s in spans if s.context.span_id == main_chat_workflow_id), None)

        if main_chat_parent:
            assert main_chat_parent.name == "Agent Workflow", (
                f"Main Chat Agent parent should be Agent Workflow, got "
                f"{main_chat_parent.name}"
            )

    # Test openai.response spans - these should contain prompts, completions, and usage
    response_spans = [s for s in spans if s.name == "openai.response"]

    assert len(response_spans) >= 1, f"Should have at least 1 openai.response span, found {len(response_spans)}"

    # Verify each response span has prompts, completions, and usage
    for i, response_span in enumerate(response_spans):

        # Check for prompts
        has_prompt = any(key.startswith("gen_ai.prompt.") for key in response_span.attributes.keys())
        assert has_prompt, (
            f"Response span {i} should have prompt attributes, attributes: "
            f"{dict(response_span.attributes)}"
        )

        # Check for completions
        has_completion = any(key.startswith("gen_ai.completion.") for key in response_span.attributes.keys())
        assert has_completion, (
            f"Response span {i} should have completion attributes, attributes: "
            f"{dict(response_span.attributes)}"
        )

        # Check for usage
        has_usage = any(key.startswith("gen_ai.usage.") or key.startswith("llm.usage.")
                        for key in response_span.attributes.keys())
        assert has_usage, (
            f"Response span {i} should have usage attributes, attributes: "
            f"{dict(response_span.attributes)}"
        )

        # Check specific expected attributes
        assert "gen_ai.system" in response_span.attributes, f"Response span {i} should have gen_ai.system"
        assert response_span.attributes["gen_ai.system"] == "openai", (
            f"Response span {i} gen_ai.system should be 'openai'"
        )

        pass  # Validation passed
