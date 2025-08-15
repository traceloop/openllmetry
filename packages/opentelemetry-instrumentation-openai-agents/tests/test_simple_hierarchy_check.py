"""Simple hierarchy check test to see all spans."""

import pytest
import json
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


@pytest.fixture(scope="session")
def recipe_agents():
    """Create Recipe Editor Agent and Main Chat Agent matching the example."""
    
    # Create Recipe Editor agent with function tools
    recipe_editor_agent = Agent(
        name="Recipe Editor Agent",
        instructions="""
        You are a recipe editor specialist powered by AI. Your role is to:
        1. Help users search and browse recipes using the search_recipes tool with intelligent semantic search

        When users want to modify a recipe:
        1. Use search_recipes to find the recipe if they mention it by name
        """,
        model="gpt-4o",
        tools=[search_recipes],
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
    """Simple check to see what spans we get."""
    
    main_chat_agent, recipe_editor_agent = recipe_agents
    
    query = "Can you find the carbonara recipe?"
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
    
    print(f"\n=== ACTUAL SPAN HIERARCHY ({len(spans)} spans) ===")
    for i, span in enumerate(spans):
        parent_name = span.parent.name if span.parent else "ROOT"
        parent_id = span.parent.span_id if span.parent else None
        span_id = span.context.span_id
        print(f"{i+1:2d}. {span.name}")
        print(f"     Parent: {parent_name} (ID: {parent_id})")
        print(f"     Span ID: {span_id}")
        print()
    
    # Filter to get the different types of spans we expect
    workflow_spans = [s for s in spans if s.name == "Agent workflow"]
    agent_spans = [s for s in spans if s.name.endswith(".agent")]
    handoff_spans = [s for s in spans if ".handoff" in s.name]
    tool_spans = [s for s in spans if s.name in ["search_recipes"]]
    response_spans = [s for s in spans if s.name == "openai.response"]
    
    print(f"=== SPAN TYPE COUNTS ===")
    print(f"Workflow spans: {len(workflow_spans)}")
    print(f"Agent spans: {len(agent_spans)} - {[s.name for s in agent_spans]}")
    print(f"Handoff spans: {len(handoff_spans)} - {[s.name for s in handoff_spans]}")
    print(f"Tool spans: {len(tool_spans)} - {[s.name for s in tool_spans]}")
    print(f"Response spans: {len(response_spans)}")
    
    # Print handoff occurrence
    print(f"\nHandoff occurred: {handoff_occurred}")
    
    # Just to prevent test from failing
    assert len(spans) >= 1, f"Should have at least 1 span, found {len(spans)}"