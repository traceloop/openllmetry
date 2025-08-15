"""Simple test to see the actual hierarchy structure."""

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
async def test_simple_hierarchy_check(exporter, recipe_agents):
    """Simple test to see actual hierarchy."""
    
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
    
    # Continue with recipe editor agent if handoff occurred
    if handoff_occurred:
        recipe_messages = [{"role": "user", "content": query}]
        recipe_runner = Runner().run_streamed(starting_agent=recipe_editor_agent, input=recipe_messages)
        
        async for event in recipe_runner.stream_events():
            pass  # Process all events
    
    spans = exporter.get_finished_spans()
    
    # Write hierarchy to a fixed file for debugging
    hierarchy_file = "/tmp/span_hierarchy.txt"
    with open(hierarchy_file, 'w') as f:
        f.write(f"=== ACTUAL SPAN HIERARCHY ({len(spans)} spans) ===\n\n")
        for i, span in enumerate(spans):
            parent_name = span.parent.name if span.parent else "ROOT"
            parent_id = span.parent.span_id if span.parent else None
            span_id = span.context.span_id
            f.write(f"{i+1:2d}. {span.name}\n")
            f.write(f"     Parent: {parent_name} (ID: {parent_id})\n")
            f.write(f"     Span ID: {span_id}\n\n")
        
        # Also write span type counts
        workflow_spans = [s for s in spans if s.name == "Agent workflow"]
        agent_spans = [s for s in spans if s.name.endswith(".agent")]
        handoff_spans = [s for s in spans if ".handoff" in s.name]
        tool_spans = [s for s in spans if s.name in ["search_recipes", "plan_and_apply_recipe_modifications"]]
        response_spans = [s for s in spans if s.name == "openai.response"]
        
        f.write(f"=== SPAN TYPE COUNTS ===\n")
        f.write(f"Workflow spans: {len(workflow_spans)}\n")
        f.write(f"Agent spans: {len(agent_spans)} - {[s.name for s in agent_spans]}\n")
        f.write(f"Handoff spans: {len(handoff_spans)} - {[s.name for s in handoff_spans]}\n")
        f.write(f"Tool spans: {len(tool_spans)} - {[s.name for s in tool_spans]}\n")
        f.write(f"Response spans: {len(response_spans)}\n")
        f.write(f"Handoff occurred: {handoff_occurred}\n")
    
    print(f"Found {len(spans)} spans")
    # Make test pass regardless
    assert True