"""Test recipe agents hierarchy matching the expected trace structure."""

import pytest
import json
from typing import Dict, List
from dataclasses import dataclass
from pydantic import BaseModel
from agents import Agent, function_tool, RunContextWrapper, Runner
from opentelemetry.trace import StatusCode
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues


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
    
    # Write hierarchy to a fixed file for debugging
    hierarchy_file = "/tmp/span_hierarchy.txt"
    with open(hierarchy_file, 'w') as f:
        f.write(f"=== ACTUAL SPAN HIERARCHY ({len(spans)} spans) ===\n")
        for i, span in enumerate(spans):
            # Find parent span name by searching spans for matching span context
            parent_name = "ROOT"
            parent_id = None
            if span.parent:
                parent_id = span.parent.span_id
                for potential_parent in spans:
                    if potential_parent.context.span_id == span.parent.span_id:
                        parent_name = potential_parent.name
                        break
            
            span_id = span.context.span_id
            f.write(f"{i+1:2d}. {span.name}\n")
            f.write(f"     Parent: {parent_name} (ID: {parent_id})\n")
            f.write(f"     Span ID: {span_id}\n")
            f.write("\n")
        
        # Also write span type counts
        workflow_spans = [s for s in spans if s.name == "Agent workflow"]
        agent_spans = [s for s in spans if s.name.endswith(".agent")]
        handoff_spans = [s for s in spans if ".handoff" in s.name]
        tool_spans = [s for s in spans if s.name in ["search_recipes", "plan_and_apply_recipe_modifications"]]
        response_spans = [s for s in spans if s.name == "openai.response"]
        
        f.write(f"\n=== SPAN TYPE COUNTS ===\n")
        f.write(f"Workflow spans: {len(workflow_spans)}\n")
        f.write(f"Agent spans: {len(agent_spans)} - {[s.name for s in agent_spans]}\n")
        f.write(f"Handoff spans: {len(handoff_spans)} - {[s.name for s in handoff_spans]}\n")
        f.write(f"Tool spans: {len(tool_spans)} - {[s.name for s in tool_spans]}\n")
        f.write(f"Response spans: {len(response_spans)}\n")
        f.write(f"Handoff occurred: {handoff_occurred}\n")
    
    
    
    # Verify we have the expected root span
    assert len(workflow_spans) == 1, f"Should have exactly 1 Agent workflow span, found {len(workflow_spans)}"
    workflow_span = workflow_spans[0]
    assert workflow_span.parent is None, "Agent workflow should be root span"
    
    # Verify we have both agent spans
    main_chat_spans = [s for s in agent_spans if "Main Chat Agent" in s.name]
    recipe_editor_spans = [s for s in agent_spans if "Recipe Editor Agent" in s.name]
    
    assert len(main_chat_spans) == 1, f"Should have exactly 1 Main Chat Agent span, found {len(main_chat_spans)}"
    assert len(recipe_editor_spans) == 1, f"Should have exactly 1 Recipe Editor Agent span, found {len(recipe_editor_spans)}"
    
    main_chat_span = main_chat_spans[0] 
    recipe_editor_span = recipe_editor_spans[0]
    
    # Verify agent spans are children of workflow
    assert main_chat_span.parent is not None, "Main Chat Agent should have parent"
    assert recipe_editor_span.parent is not None, "Recipe Editor Agent should have parent"
    assert main_chat_span.parent.span_id == workflow_span.context.span_id, "Main Chat Agent should be child of workflow"
    assert recipe_editor_span.parent.span_id == workflow_span.context.span_id, "Recipe Editor Agent should be child of workflow"
    
    # Verify handoff span exists and is child of main chat agent - THIS IS THE KEY FIX
    assert len(handoff_spans) >= 1, f"Should have at least 1 handoff span, found {len(handoff_spans)}"
    handoff_span = handoff_spans[0]
    assert handoff_span.parent is not None, "Handoff span should have parent"
    assert handoff_span.parent.span_id == main_chat_span.context.span_id, "Handoff should be child of Main Chat Agent"
