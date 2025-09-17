import asyncio
import json
from typing import Dict, List
from dataclasses import dataclass
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
from traceloop.sdk import Traceloop

from agents import Agent, function_tool, RunContextWrapper, Runner, ToolCallOutputItem
from openai.types.responses import (
    ResponseTextDeltaEvent,
    ResponseOutputItemAddedEvent,
    ResponseFunctionToolCall,
    ResponseOutputText,
    ResponseOutputRefusal,
)

load_dotenv()

Traceloop.init(
    app_name="openai-agents-demo",
    disable_batch=False
)


class PlannedChange(BaseModel):
    type: str
    description: str
    target: str


class ModificationPlan(BaseModel):
    recipe_id: str
    recipe_name: str
    modification_request: str
    planned_changes: List[PlannedChange]
    confidence: str


class PlanResponse(BaseModel):
    status: str
    message: str
    plan: ModificationPlan | None = None
    available_recipes: List[str] | None = None


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
    available_recipes: List[str] | None = None


class SearchResponse(BaseModel):
    status: str
    message: str
    recipes: Dict[str, Recipe] | None = None
    recipe_count: int | None = None
    query: str | None = None


class ModifiedRecipeData(BaseModel):
    id: str
    name: str
    ingredients: List[str]
    instructions: List[str]
    prep_time: str
    cook_time: str
    servings: int


class RecipeModificationResult(BaseModel):
    modified_recipe: ModifiedRecipeData
    changes_made: List[str]
    modification_reasoning: str


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
    },
    "chicken_tikka_masala": {
        "id": "chicken_tikka_masala",
        "name": "Chicken Tikka Masala",
        "ingredients": [
            "500g chicken breast, cubed",
            "200ml plain yogurt",
            "2 tsp garam masala",
            "1 tsp turmeric",
            "1 onion, diced",
            "3 cloves garlic, minced",
            "1 inch ginger, grated",
            "400ml coconut milk",
            "400g canned tomatoes",
            "2 tbsp tomato paste",
            "1 tsp cumin",
            "1 tsp coriander",
            "Salt and pepper",
            "Fresh cilantro",
        ],
        "instructions": [
            "Marinate chicken in yogurt, garam masala, and turmeric for 30 minutes",
            "Grill or pan-fry chicken pieces until cooked through",
            "Sauté onion until golden, add garlic and ginger",
            "Add spices and cook for 1 minute",
            "Add tomatoes and tomato paste, simmer 10 minutes",
            "Add coconut milk and cooked chicken",
            "Simmer for 15 minutes until thick",
            "Garnish with cilantro and serve with rice",
        ],
        "prep_time": "45 minutes",
        "cook_time": "30 minutes",
        "servings": 4,
    },
    "chocolate_chip_cookies": {
        "id": "chocolate_chip_cookies",
        "name": "Classic Chocolate Chip Cookies",
        "ingredients": [
            "225g butter, softened",
            "200g brown sugar",
            "100g white sugar",
            "2 large eggs",
            "1 tsp vanilla extract",
            "300g plain flour",
            "1 tsp baking soda",
            "1 tsp salt",
            "300g chocolate chips",
        ],
        "instructions": [
            "Preheat oven to 190°C (375°F)",
            "Cream butter and both sugars until light and fluffy",
            "Beat in eggs and vanilla",
            "Mix flour, baking soda, and salt in separate bowl",
            "Gradually add dry ingredients to wet ingredients",
            "Fold in chocolate chips",
            "Drop rounded tablespoons onto baking sheet",
            "Bake for 9-11 minutes until golden brown",
            "Cool on baking sheet for 5 minutes before transferring",
        ],
        "prep_time": "15 minutes",
        "cook_time": "10 minutes",
        "servings": 24,
    },
    "beef_stir_fry": {
        "id": "beef_stir_fry",
        "name": "Beef and Vegetable Stir Fry",
        "ingredients": [
            "500g beef sirloin, sliced thin",
            "2 tbsp vegetable oil",
            "1 bell pepper, sliced",
            "1 onion, sliced",
            "200g broccoli florets",
            "2 carrots, sliced",
            "3 cloves garlic, minced",
            "2 tbsp soy sauce",
            "1 tbsp oyster sauce",
            "1 tsp sesame oil",
            "1 tsp cornstarch",
            "2 tbsp water",
            "Green onions for garnish",
        ],
        "instructions": [
            "Heat wok or large pan over high heat",
            "Add oil and beef, stir-fry for 2-3 minutes",
            "Remove beef and set aside",
            "Add vegetables to pan, stir-fry for 3-4 minutes",
            "Add garlic and cook for 30 seconds",
            "Return beef to pan",
            "Mix soy sauce, oyster sauce, sesame oil, cornstarch and water",
            "Add sauce to pan, stir for 1 minute until thickened",
            "Garnish with green onions and serve with rice",
        ],
        "prep_time": "15 minutes",
        "cook_time": "8 minutes",
        "servings": 4,
    },
    "caesar_salad": {
        "id": "caesar_salad",
        "name": "Classic Caesar Salad",
        "ingredients": [
            "2 heads romaine lettuce, chopped",
            "1/2 cup mayonnaise",
            "2 tbsp lemon juice",
            "2 cloves garlic, minced",
            "1 tsp Dijon mustard",
            "1 tsp Worcestershire sauce",
            "1/2 cup Parmesan cheese, grated",
            "2 anchovy fillets, minced (optional)",
            "1 cup croutons",
            "Black pepper",
            "Salt",
        ],
        "instructions": [
            "Wash and dry romaine lettuce thoroughly",
            "Whisk mayonnaise, lemon juice, garlic, and mustard",
            "Add Worcestershire sauce and anchovies if using",
            "Season dressing with salt and pepper",
            "Toss lettuce with dressing",
            "Add most of the Parmesan cheese and toss",
            "Top with croutons and remaining cheese",
            "Serve immediately",
        ],
        "prep_time": "15 minutes",
        "cook_time": "0 minutes",
        "servings": 4,
    },
}


@dataclass
class ChatContext:
    """Standalone context for the chat application."""

    conversation_history: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []


@function_tool
async def plan_and_apply_recipe_modifications(
    cw: RunContextWrapper[ChatContext], recipe: Recipe, modification_request: str
) -> EditResponse:
    """
    Plan modifications to a recipe based on user request and apply them.

    Args:
        recipe_name: Name of the recipe to modify
        modification_request: Description of the desired changes

    Returns:
        Dictionary containing planned modifications and applied modifications
    """
    print(f"Planning and applying modifications for recipe: {recipe.name}")
    print(f"Modification request: {modification_request}")

    client = openai.OpenAI()

    modification_prompt = f"""
You are an expert chef and recipe developer. You need to modify an existing recipe based on a user's request.

Original Recipe:
Name: {recipe.name}
Ingredients: {json.dumps(recipe.ingredients, indent=2)}
Instructions: {json.dumps(recipe.instructions, indent=2)}
Prep Time: {recipe.prep_time}
Cook Time: {recipe.cook_time}
Servings: {recipe.servings}

User's Modification Request: "{modification_request}"

Please create a modified version of this recipe that incorporates the user's request. Consider:
- Ingredient substitutions (vegetarian/vegan, gluten-free, allergies, etc.)
- Scaling (doubling, halving, different serving sizes)
- Cooking method changes (faster, slower, different techniques)
- Flavor modifications (spicier, less salty, sweeter, etc.)
- Dietary restrictions and preferences
- Seasonal ingredient swaps
- Nutritional improvements

Make sure all ingredients have proper quantities and units. Keep the essence of the original recipe while making
the requested modifications. Be creative but practical.

Provide a clear list of what changes you made and explain your reasoning for the modifications.
"""

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert chef and recipe developer who modifies recipes based on user requests.",
            },
            {"role": "user", "content": modification_prompt},
        ],
        temperature=0.3,
        max_tokens=1500,
        response_format=RecipeModificationResult,
    )

    modification_result = response.choices[0].message.parsed

    modified_recipe_data = modification_result.modified_recipe.model_dump()
    changes_made = modification_result.changes_made
    reasoning = modification_result.modification_reasoning
    required_fields = ["name", "ingredients", "instructions"]

    for field in required_fields:
        if field not in modified_recipe_data:
            print(f"\n[LLM response missing required field: {field}]\n")
            return EditResponse(
                status="error", message=f"LLM response missing required field: {field}"
            )

    modified_recipe_data["id"] = recipe.id
    if "prep_time" not in modified_recipe_data:
        modified_recipe_data["prep_time"] = recipe.prep_time
    if "cook_time" not in modified_recipe_data:
        modified_recipe_data["cook_time"] = recipe.cook_time
    if "servings" not in modified_recipe_data:
        modified_recipe_data["servings"] = recipe.servings

    GLOBAL_RECIPE_CONTEXT[recipe.id] = modified_recipe_data

    return EditResponse(
        status="success",
        message=f'Successfully modified {modified_recipe_data["name"]} using AI. {reasoning}',
        modified_recipe=Recipe(**modified_recipe_data),
        changes_made=changes_made,
        original_recipe=recipe,
    )


@function_tool
async def search_recipes(
    cw: RunContextWrapper[ChatContext], query: str = ""
) -> SearchResponse:
    """
    Search and browse recipes in the database.

    Args:
        query: Search term to filter recipes (optional)

    Returns:
        Dictionary containing matching recipes
    """
    print(f"Searching recipes for query: '{query}'")

    try:
        recipe_context = GLOBAL_RECIPE_CONTEXT

        if not query:
            recipes_dict = {k: Recipe(**v) for k, v in recipe_context.items()}
            return SearchResponse(
                status="success",
                message=f"Found {len(recipe_context)} recipes in database",
                recipes=recipes_dict,
                recipe_count=len(recipe_context),
            )

        client = openai.OpenAI()
        recipe_summaries = []

        for recipe_id, recipe in recipe_context.items():
            summary = {
                "id": recipe_id,
                "name": recipe["name"],
                "ingredients": recipe["ingredients"][
                    :5
                ],  # First 5 ingredients for brevity
                "cuisine_type": "various",  # Could be enhanced with actual cuisine classification
                "prep_time": recipe.get("prep_time", "Unknown"),
                "cook_time": recipe.get("cook_time", "Unknown"),
                "servings": recipe.get("servings", "Unknown"),
            }
            recipe_summaries.append(summary)

        search_prompt = f"""
You are a recipe search expert. Given a user's search query and a list of available recipes,
identify which recipes are most relevant to the user's request.

User's search query: "{query}"

Available recipes:
{json.dumps(recipe_summaries, indent=2)}

Please analyze the query and return a JSON response with the following format:
{{
    "relevant_recipe_ids": ["recipe_id1", "recipe_id2", ...],
    "reasoning": "Brief explanation of why these recipes match the query"
}}

Consider:
- Ingredient matches (exact or similar)
- Cuisine type preferences
- Cooking method preferences
- Dietary restrictions (vegetarian, vegan, gluten-free, etc.)
- Meal type (breakfast, lunch, dinner, dessert)
- Difficulty level or time preferences
- Flavor profiles (spicy, sweet, savory, etc.)

Return only the JSON response, no additional text.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful recipe search assistant that returns only valid JSON.",
                },
                {"role": "user", "content": search_prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        )

        llm_response = response.choices[0].message.content.strip()

        try:
            search_result = json.loads(llm_response)
            relevant_ids = search_result.get("relevant_recipe_ids", [])
            reasoning = search_result.get("reasoning", "LLM analysis")
        except json.JSONDecodeError:
            print(
                "Failed to parse LLM response as JSON, falling back to keyword search"
            )

            query_lower = query.lower()
            relevant_ids = []
            for recipe_id, recipe in recipe_context.items():
                if query_lower in recipe["name"].lower() or any(
                    query_lower in ingredient.lower()
                    for ingredient in recipe["ingredients"]
                ):
                    relevant_ids.append(recipe_id)
            reasoning = "Keyword-based fallback search"

        matching_recipes = {}
        for recipe_id in relevant_ids:
            if recipe_id in recipe_context:
                matching_recipes[recipe_id] = recipe_context[recipe_id]

        recipes_dict = {k: Recipe(**v) for k, v in matching_recipes.items()}

        message = (
            f'Found {len(matching_recipes)} recipes matching "{query}". {reasoning}'
        )

        return SearchResponse(
            status="success",
            message=message,
            recipes=recipes_dict,
            recipe_count=len(matching_recipes),
            query=query,
        )

    except Exception as e:
        print(f"Error searching recipes: {str(e)}")
        return SearchResponse(
            status="error", message=f"Failed to search recipes: {str(e)}"
        )


class RecipeEditorAgent(Agent[ChatContext]):
    """Specialized agent for recipe editing and management tasks with function tools."""

    def __init__(self, model: str = "gpt-4o"):
        super().__init__(
            name="Recipe Editor Agent",
            instructions="""
            You are a recipe editor specialist powered by AI. Your role is to:
            1. Help users search and browse recipes using the search_recipes tool with intelligent semantic search
            2. Modify recipes using AI-powered analysis with the
               plan_and_apply_recipe_modifications tool

            Your capabilities:
            - Search recipes using natural language queries (e.g., "healthy dinner", "quick breakfast",
              "spicy vegetarian")
            - Intelligently modify recipes based on dietary restrictions, preferences, scaling, and cooking methods
            - Make complex ingredient substitutions and cooking technique adjustments
            - Provide detailed explanations of changes and reasoning

            When users want to modify a recipe:
            1. Use search_recipes to find the recipe if they mention it by name
            2. Use plan_and_apply_recipe_modifications to intelligently modify the recipe using AI

            The AI system will handle complex modifications like:
            - Dietary conversions (vegetarian, vegan, gluten-free, keto, etc.)
            - Scaling recipes up or down
            - Flavor profile changes (spicier, less salty, sweeter)
            - Cooking method improvements (faster, healthier, easier)
            - Ingredient substitutions based on availability or preferences

            Always explain the changes made and provide helpful cooking tips.
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


async def handle_runner_stream(runner: "Runner"):
    """Utility to iterate over a Runner's event stream and
    print assistant deltas, tool invocations, and tool outputs.

    Args:
        runner: The streaming `Runner` instance returned by
            `Runner().run_streamed(...)`.

    Returns:
        The raw handoff information object if a hand-off occurs, otherwise
        ``None``.
    """

    handoff_info = None

    async for event in runner.stream_events():
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
            elif isinstance(event.data, ResponseOutputItemAddedEvent):
                if isinstance(event.data.item, ResponseFunctionToolCall):
                    print(f"\n[Calling tool: {event.data.item.name}]\n")
        elif event.type == "run_item_stream_event":
            if event.name == "tool_output" and isinstance(
                event.item, ToolCallOutputItem
            ):
                raw_item = event.item.raw_item
                content = (
                    raw_item.get("content")
                    if isinstance(raw_item, dict)
                    else getattr(raw_item, "content", "")
                )
                if content:
                    print(f"\n[Tool output: {content}]\n", end="", flush=True)

            elif event.name == "message_output_created":
                raw_item = event.item.raw_item

                role = getattr(raw_item, "role", None)
                if role is None and isinstance(raw_item, dict):
                    role = raw_item.get("role")

                if role == "assistant":
                    content_parts = []
                    for part in getattr(raw_item, "content", []):
                        if isinstance(part, ResponseOutputText):
                            content_parts.append(part.text)
                        elif isinstance(part, ResponseOutputRefusal):
                            content_parts.append(part.refusal)
                    if content_parts:
                        print("".join(content_parts), end="", flush=True)

            elif event.name == "handoff_occurred":
                handoff_info = event.item.raw_item
                print("\n[Handed off to Recipe Editor Agent]\n")

    print()
    return handoff_info


async def run_streaming_chat(user_input: str):
    """
    Run the streaming chat application with agent handoffs.
    """
    print("Starting Streaming Chat Application with Agent Handoffs")
    print("=" * 60)

    chat_ctx = ChatContext(conversation_history=[])

    recipe_editor_agent = RecipeEditorAgent()
    main_chat_agent = MainChatAgent(recipe_editor_agent=recipe_editor_agent)

    print("Agents initialized successfully")
    print("Try asking about recipes, cooking, or ingredient modifications")
    print("Example: 'Can you show me some recipes?' or 'Make the carbonara vegetarian'")
    print("Type 'quit' to exit")
    print("-" * 60)

    print(f"\nYou: {user_input}", end="", flush=True)
    chat_ctx.conversation_history.append({"role": "user", "content": user_input})

    print("\nAssistant: ", end="", flush=True)

    messages = [{"role": "user", "content": user_input}]
    main_runner = Runner().run_streamed(starting_agent=main_chat_agent, input=messages)
    handoff_info = await handle_runner_stream(main_runner)

    if handoff_info and "recipe" in str(handoff_info).lower():
        recipe_messages = [{"role": "user", "content": user_input}]
        recipe_runner = Runner().run_streamed(
            starting_agent=recipe_editor_agent, input=recipe_messages
        )
        await handle_runner_stream(recipe_runner)

    print(f"\n{'='*60}")
    print("✅ OpenAI Agents demo completed successfully!")
    print("🔍 Spans are being captured by the OpenTelemetry instrumentation")
    print(f"{'='*60}")


if __name__ == "__main__":
    """
    Main entry point for the streaming chat application.

    This demonstrates:
    1. MainChatAgent handling general conversation
    2. Handoff to RecipeEditorAgent for specialized tasks
    3. RecipeEditorAgent using function tools in sequence
    4. Streaming responses back to the user
    """
    print("Demo: OpenAI Agents SDK with Handoffs and Function Tools")
    print("Use case: Recipe Management and Editing")

    user_input = "Can you edit the carbonara recipe to be vegetarian?"

    asyncio.run(run_streaming_chat(user_input))
