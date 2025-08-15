"""Debug version of recipe agents hierarchy test."""

import pytest
from typing import List
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
    recipes: dict | None = None
    recipe_count: int | None = None
    query: str | None = None


@function_tool
async def search_recipes(
    cw: RunContextWrapper, query: str = ""
) -> SearchResponse:
    """Search and browse recipes in the database."""
    # Mock response
    if "carbonara" in query.lower():
        sample_recipe = Recipe(
            id="spaghetti_carbonara",
            name="Spaghetti Carbonara",
            ingredients=["400g spaghetti", "200g pancetta"],
            instructions=["Cook spaghetti", "Add pancetta"],
            prep_time="10 minutes",
            cook_time="15 minutes",
            servings=4,
        )
        return SearchResponse(
            status="success",
            message=f"Found 1 recipe matching '{query}'",
            recipes={"spaghetti_carbonara": sample_recipe},
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


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_debug_recipe_hierarchy(exporter):
    """Debug test to see recipe agents hierarchy."""
    
    # Create Recipe Editor agent with function tools
    recipe_editor_agent = Agent(
        name="Recipe Editor Agent",
        instructions="You help with recipe searches.",
        model="gpt-4o",
        tools=[search_recipes],
    )
    
    # Create Main Chat agent that can hand off to Recipe Editor
    main_chat_agent = Agent(
        name="Main Chat Agent",
        instructions="You handle cooking requests and transfer to recipe editor agent when needed.",
        model="gpt-4o",
        handoffs=[recipe_editor_agent],
    )
    
    query = "Can you find carbonara recipe?"
    messages = [{"role": "user", "content": query}]
    
    # Run the main chat agent which should handoff to recipe editor
    main_runner = Runner().run_streamed(starting_agent=main_chat_agent, input=messages)
    
    handoff_occurred = False
    async for event in main_runner.stream_events():
        if event.type == "run_item_stream_event":
            if "handoff" in event.name.lower():
                handoff_occurred = True
                print(f"DEBUG: Handoff event detected: {event.name}")
    
    # Continue with recipe editor agent if handoff occurred
    if handoff_occurred:
        print("DEBUG: Running recipe editor agent...")
        recipe_messages = [{"role": "user", "content": query}]
        recipe_runner = Runner().run_streamed(starting_agent=recipe_editor_agent, input=recipe_messages)
        
        async for event in recipe_runner.stream_events():
            if event.type == "run_item_stream_event":
                print(f"DEBUG: Recipe editor event: {event.name}")
    
    spans = exporter.get_finished_spans()
    
    print(f"\n=== DEBUG: Found {len(spans)} spans ===")
    for i, span in enumerate(spans):
        parent_name = span.parent.name if span.parent else "ROOT"
        parent_id = span.parent.span_id if span.parent else None
        print(f"{i+1}. {span.name} (parent: {parent_name}, parent_id: {parent_id})")
        
        # Show attributes
        attrs = dict(span.attributes)
        if attrs:
            print(f"   Attributes: {attrs}")
    
    # Check for specific span types
    workflow_spans = [s for s in spans if s.name == "Agent workflow"]
    agent_spans = [s for s in spans if s.name.endswith(".agent")]
    handoff_spans = [s for s in spans if ".handoff" in s.name]
    tool_spans = [s for s in spans if s.name in ["search_recipes"]]
    response_spans = [s for s in spans if s.name == "openai.response"]
    
    print(f"\n=== SPAN COUNTS ===")
    print(f"Workflow spans: {len(workflow_spans)}")
    print(f"Agent spans: {len(agent_spans)} - {[s.name for s in agent_spans]}")
    print(f"Handoff spans: {len(handoff_spans)} - {[s.name for s in handoff_spans]}")
    print(f"Tool spans: {len(tool_spans)} - {[s.name for s in tool_spans]}")
    print(f"Response spans: {len(response_spans)}")
    
    print(f"\nHandoff occurred: {handoff_occurred}")
    
    # Just to prevent test from failing, add a simple assertion
    assert len(spans) >= 1, f"Should have at least 1 span, found {len(spans)}"