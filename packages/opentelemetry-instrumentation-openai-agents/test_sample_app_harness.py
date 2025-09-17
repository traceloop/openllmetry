#!/usr/bin/env python3
"""
Test Sample App Harness

A controlled version of the sample app for recording and testing agent name propagation.
This eliminates external dependencies and allows us to focus on span generation.
"""

import asyncio
import json
from typing import Dict, List
from dataclasses import dataclass
from pydantic import BaseModel

from agents import Agent, function_tool, Runner, RunContextWrapper


# Sample app models (simplified for testing)
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


class SearchResponse(BaseModel):
    status: str
    message: str
    recipes: Dict[str, Recipe] | None = None
    recipe_count: int | None = None


@dataclass
class ChatContext:
    conversation_history: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []


# Test recipe data
CARBONARA_RECIPE = Recipe(
    id="spaghetti_carbonara",
    name="Spaghetti Carbonara",
    ingredients=[
        "400g spaghetti", "200g pancetta", "4 large eggs",
        "100g Pecorino Romano cheese", "2 cloves garlic", "Black pepper", "Salt"
    ],
    instructions=[
        "Cook spaghetti in salted boiling water until al dente",
        "Dice pancetta and cook in a large pan until crispy",
        "Whisk eggs with grated cheese and black pepper",
        "Drain pasta, reserving 1 cup pasta water",
        "Add hot pasta to the pan with pancetta",
        "Remove from heat, add egg mixture, toss quickly",
        "Add pasta water if needed to create creamy sauce",
        "Serve immediately with extra cheese"
    ],
    prep_time="10 minutes",
    cook_time="15 minutes",
    servings=4
)


@function_tool
async def search_recipes_harness(
    cw: RunContextWrapper[ChatContext], query: str = ""
) -> SearchResponse:
    """Test version of search_recipes that returns controlled results."""
    print(f"🔍 Searching recipes for: '{query}'")

    if "carbonara" in query.lower():
        return SearchResponse(
            status="success",
            message=f'Found 1 recipe matching "{query}"',
            recipes={"spaghetti_carbonara": CARBONARA_RECIPE},
            recipe_count=1
        )

    return SearchResponse(
        status="success",
        message="No recipes found",
        recipes={},
        recipe_count=0
    )


@function_tool
async def modify_recipe_harness(
    cw: RunContextWrapper[ChatContext], recipe: Recipe, modification: str
) -> EditResponse:
    """Test version of recipe modification that returns controlled results."""
    print(f"🔧 Modifying recipe: {recipe.name} -> {modification}")

    if "vegetarian" in modification.lower():
        # Create vegetarian version
        vegetarian_recipe = Recipe(
            id=recipe.id,
            name="Vegetarian Carbonara",
            ingredients=[
                "400g spaghetti", "200g mushrooms", "4 large eggs",
                "100g Pecorino Romano cheese", "2 cloves garlic", "Black pepper", "Salt"
            ],
            instructions=[
                "Cook spaghetti in salted boiling water until al dente",
                "Dice mushrooms and cook in a large pan until golden",
                "Whisk eggs with grated cheese and black pepper",
                "Drain pasta, reserving 1 cup pasta water",
                "Add hot pasta to the pan with mushrooms",
                "Remove from heat, add egg mixture, toss quickly",
                "Add pasta water if needed to create creamy sauce",
                "Serve immediately with extra cheese"
            ],
            prep_time=recipe.prep_time,
            cook_time=recipe.cook_time,
            servings=recipe.servings
        )

        return EditResponse(
            status="success",
            message="Successfully converted carbonara to vegetarian version",
            modified_recipe=vegetarian_recipe,
            changes_made=["Replaced pancetta with mushrooms"]
        )

    return EditResponse(
        status="error",
        message="Could not apply modification"
    )


class RecipeEditorAgentHarness(Agent[ChatContext]):
    """Test harness version of Recipe Editor Agent."""

    def __init__(self):
        super().__init__(
            name="Recipe Editor Agent",
            instructions="""
            You are a recipe editor. Use search_recipes_harness to find recipes
            and modify_recipe_harness to modify them based on user requests.

            When a user asks to edit a recipe:
            1. First search for the recipe
            2. Then modify it according to their request
            """,
            model="gpt-4o",
            tools=[search_recipes_harness, modify_recipe_harness]
        )


class MainChatAgentHarness(Agent[ChatContext]):
    """Test harness version of Main Chat Agent."""

    def __init__(self, recipe_agent):
        super().__init__(
            name="Main Chat Agent",
            instructions="""
            You are a helpful assistant that routes recipe-related requests
            to the recipe editor agent. When users mention recipes, cooking,
            or food modifications, hand off to the recipe editor.
            """,
            model="gpt-4o",
            handoffs=[recipe_agent]
        )


async def run_harness_workflow():
    """Run the test harness workflow that mimics the sample app."""

    print("🎬 Running Sample App Harness Workflow")
    print("=" * 50)

    # Create agents
    recipe_agent = RecipeEditorAgentHarness()
    main_agent = MainChatAgentHarness(recipe_agent)

    # Test message that should trigger the full workflow
    user_input = "Can you edit the carbonara recipe to be vegetarian?"
    print(f"👤 User: {user_input}")

    try:
        # Step 1: Main agent should decide to hand off
        print("\n🤖 Running Main Chat Agent...")
        messages = [{"role": "user", "content": user_input}]

        # Use threading to avoid event loop conflicts
        import threading

        def run_main_agent():
            try:
                result1 = Runner.run_sync(main_agent, messages)
                print(f"Main agent result: {result1}")
            except Exception as e:
                print(f"Main agent error (expected): {e}")

        def run_recipe_agent():
            try:
                result2 = Runner.run_sync(recipe_agent, messages)
                print(f"Recipe agent result: {result2}")
            except Exception as e:
                print(f"Recipe agent error (expected): {e}")

        # Run main agent in thread
        thread1 = threading.Thread(target=run_main_agent)
        thread1.start()
        thread1.join()

        # Step 2: Recipe editor agent should search and modify
        print("\n🍝 Running Recipe Editor Agent...")

        thread2 = threading.Thread(target=run_recipe_agent)
        thread2.start()
        thread2.join()

        print("\n✅ Workflow completed!")

    except Exception as e:
        print(f"❌ Workflow error: {e}")


def run_simple_single_agent_test():
    """Run a simpler single agent test for comparison."""

    print("\n🧪 Running Simple Single Agent Test")
    print("=" * 50)

    # Create a simple agent
    simple_agent = Agent(
        name="Simple Test Agent",
        instructions="You are a simple test agent.",
        model="gpt-4o"
    )

    try:
        messages = [{"role": "user", "content": "Hello, simple agent!"}]

        # Use asyncio to run properly
        import threading
        result = None
        error = None

        def run_in_thread():
            nonlocal result, error
            try:
                result = Runner.run_sync(simple_agent, messages)
            except Exception as e:
                error = e

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()

        if result:
            print(f"Simple agent result: {result}")
        else:
            print(f"Simple agent error (expected): {error}")

    except Exception as e:
        print(f"Simple agent error (expected): {e}")


def run_tool_agent_test():
    """Run a test with an agent that has tools."""

    print("\n🔧 Running Tool Agent Test")
    print("=" * 50)

    @function_tool
    def simple_tool(message: str) -> str:
        """A simple tool for testing."""
        return f"Tool processed: {message}"

    class ToolTestAgent(Agent):
        def __init__(self):
            super().__init__(
                name="Tool Test Agent",
                instructions="You are an agent with tools. Use the simple_tool when asked.",
                model="gpt-4o",
                tools=[simple_tool]
            )

    tool_agent = ToolTestAgent()

    try:
        messages = [{"role": "user", "content": "Use the simple tool with message 'test'"}]

        # Use threading to avoid event loop conflicts
        import threading
        result = None
        error = None

        def run_in_thread():
            nonlocal result, error
            try:
                result = Runner.run_sync(tool_agent, messages)
            except Exception as e:
                error = e

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()

        if result:
            print(f"Tool agent result: {result}")
        else:
            print(f"Tool agent error (expected): {error}")

    except Exception as e:
        print(f"Tool agent error (expected): {e}")


if __name__ == "__main__":
    """Run various test scenarios."""

    print("🧪 Sample App Test Harness")
    print("=" * 50)
    print("This harness runs controlled versions of sample app workflows")
    print("for testing agent name propagation.")
    print()

    # Run different test scenarios
    asyncio.run(run_harness_workflow())
    run_simple_single_agent_test()
    run_tool_agent_test()

    print("\n🎯 Harness execution completed!")
    print("Check the OpenTelemetry spans for agent name propagation.")