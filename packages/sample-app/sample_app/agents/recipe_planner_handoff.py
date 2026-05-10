#!/usr/bin/env python3
"""
Recipe Planner Agent (efficient baseline) using OpenAI Agents SDK.

Calibrated to score HIGH on agent_efficiency_v2:
- Announces an explicit plan listing tools in order
- Executes every committed step
- Quotes each tool's output before moving to the next
- Final answer is a coherent meal plan

Architecture: two specialized agents connected via a handoff.
  1. Recipe Researcher  — searches recipes, checks pantry, estimates cost,
                          then hands off to the composer.
  2. Meal Plan Composer — receives the researcher's findings and composes
                          the final 3-meal plan.
"""

import asyncio
import argparse
import random
import time
from typing import Dict, List
from dataclasses import dataclass
from pydantic import BaseModel
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
    app_name="recipe-planner-agent",
    disable_batch=False,
)


RECIPE_DATABASE = {
    "R-001": {
        "name": "Chickpea curry with brown rice",
        "tags": ["vegetarian", "gluten-free"],
        "pantry_pct": 0.7,
        "cost_usd": 4.50,
    },
    "R-002": {
        "name": "Sheet-pan roasted vegetables",
        "tags": ["vegetarian", "vegan", "gluten-free"],
        "pantry_pct": 0.85,
        "cost_usd": 3.20,
    },
    "R-003": {
        "name": "Lentil soup with crusty bread",
        "tags": ["vegetarian", "vegan"],
        "pantry_pct": 0.6,
        "cost_usd": 2.80,
    },
    "R-004": {
        "name": "Pasta primavera",
        "tags": ["vegetarian"],
        "pantry_pct": 0.5,
        "cost_usd": 5.10,
    },
    "R-005": {
        "name": "Black bean tacos",
        "tags": ["vegetarian", "vegan"],
        "pantry_pct": 0.65,
        "cost_usd": 4.10,
    },
}


class Recipe(BaseModel):
    recipe_id: str
    name: str
    tags: List[str]


class RecipeSearchResponse(BaseModel):
    status: str
    message: str
    recipes: List[Recipe] = []


class PantryCheck(BaseModel):
    recipe_id: str
    pantry_pct: float


class PantryCheckResponse(BaseModel):
    status: str
    message: str
    check: PantryCheck | None = None


class CostEstimate(BaseModel):
    recipe_id: str
    cost_usd: float


class CostEstimateResponse(BaseModel):
    status: str
    message: str
    estimate: CostEstimate | None = None


class MealPlan(BaseModel):
    picks: List[str]
    total_cost_usd: float
    notes: str


class MealPlanResponse(BaseModel):
    status: str
    message: str
    plan: MealPlan | None = None


@dataclass
class RecipeContext:
    conversation_history: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []


@function_tool
async def search_recipes(
    cw: RunContextWrapper[RecipeContext], constraints: List[str]
) -> RecipeSearchResponse:
    """
    Search for recipes that match dietary constraint tags.

    Args:
        constraints: A list of dietary tags (e.g., ["vegetarian", "gluten-free"]).

    Returns:
        Up to 5 recipes that match all constraints.
    """
    print(f"[Tool: search_recipes] constraints={constraints}")
    await asyncio.sleep(0.2)
    constraints_lower = {c.lower() for c in constraints}
    matches = []
    for rid, r in RECIPE_DATABASE.items():
        if constraints_lower.issubset({t.lower() for t in r["tags"]}):
            matches.append(Recipe(recipe_id=rid, name=r["name"], tags=r["tags"]))
    matches = matches[:5]
    return RecipeSearchResponse(
        status="success",
        message=f"Found {len(matches)} recipes matching {constraints}",
        recipes=matches,
    )


@function_tool
async def check_pantry(
    cw: RunContextWrapper[RecipeContext], recipe_id: str
) -> PantryCheckResponse:
    """
    Check what fraction of a recipe's ingredients are already on hand.

    Args:
        recipe_id: The recipe ID (e.g., "R-001").

    Returns:
        A pantry-on-hand percentage between 0.0 and 1.0.
    """
    print(f"[Tool: check_pantry] recipe_id={recipe_id}")
    await asyncio.sleep(0.2)
    r = RECIPE_DATABASE.get(recipe_id)
    if not r:
        return PantryCheckResponse(status="error", message=f"Recipe {recipe_id} not found")
    return PantryCheckResponse(
        status="success",
        message=f"Pantry check for {recipe_id}",
        check=PantryCheck(recipe_id=recipe_id, pantry_pct=r["pantry_pct"]),
    )


@function_tool
async def estimate_cost(
    cw: RunContextWrapper[RecipeContext], recipe_id: str
) -> CostEstimateResponse:
    """
    Estimate the dollar cost to cook a recipe given current pantry levels.

    Args:
        recipe_id: The recipe ID (e.g., "R-001").

    Returns:
        Estimated cost in USD.
    """
    print(f"[Tool: estimate_cost] recipe_id={recipe_id}")
    await asyncio.sleep(0.2)
    r = RECIPE_DATABASE.get(recipe_id)
    if not r:
        return CostEstimateResponse(status="error", message=f"Recipe {recipe_id} not found")
    return CostEstimateResponse(
        status="success",
        message=f"Cost estimate for {recipe_id}",
        estimate=CostEstimate(recipe_id=recipe_id, cost_usd=r["cost_usd"]),
    )


@function_tool
async def compose_meal_plan(
    cw: RunContextWrapper[RecipeContext], picks: List[str], notes: str = ""
) -> MealPlanResponse:
    """
    Compose a final meal plan from selected recipe IDs.

    Args:
        picks: A list of recipe IDs (e.g., ["R-001", "R-002", "R-005"]).
        notes: Optional notes describing the plan.

    Returns:
        A MealPlan with the picks, total cost, and notes.
    """
    print(f"[Tool: compose_meal_plan] picks={picks}")
    await asyncio.sleep(0.2)
    total = 0.0
    for rid in picks:
        r = RECIPE_DATABASE.get(rid)
        if r:
            total += r["cost_usd"]
    plan = MealPlan(picks=picks, total_cost_usd=round(total, 2), notes=notes)
    return MealPlanResponse(
        status="success",
        message=f"Composed meal plan with {len(picks)} recipes",
        plan=plan,
    )


class MealPlanComposerAgent(Agent[RecipeContext]):
    """Receives the researcher's findings and composes the final plan."""

    def __init__(self, model: str = "gpt-4o"):
        super().__init__(
            name="Meal Plan Composer",
            handoff_description="Composes the final 3-meal plan from researched recipes.",
            instructions="""
            You are the Meal Plan Composer. The Recipe Researcher has handed off to
            you with a set of candidate recipes plus their pantry and cost data.

            Your job:
              1. Pick exactly 3 recipes from the candidates that fit the user's
                 dietary constraints and budget.
              2. Call compose_meal_plan ONCE with those 3 recipe IDs and a short
                 notes string summarizing why they fit.
              3. QUOTE the compose_meal_plan output, then write a final answer
                 that restates the plan and confirms it fits the budget.

            Do NOT call search_recipes, check_pantry, or estimate_cost — that work
            is already done. Do NOT hand back to the researcher.
            """,
            model=model,
            tools=[compose_meal_plan],
        )


class RecipeResearcherAgent(Agent[RecipeContext]):
    """Researches candidate recipes, then hands off to the composer."""

    def __init__(self, composer: Agent[RecipeContext], model: str = "gpt-4o"):
        super().__init__(
            name="Recipe Researcher",
            handoff_description="Searches recipes and gathers pantry + cost data.",
            instructions="""
            You are the Recipe Researcher. You build the data the Meal Plan Composer
            needs to assemble a 3-meal plan within the user's dietary constraints
            and budget.

            FIRST, write a numbered plan listing exactly which tools you will call
            and in what order. The plan MUST contain these steps in this order:
              1. search_recipes
              2. check_pantry (called for each candidate recipe)
              3. estimate_cost (called for each candidate recipe)
              4. handoff to Meal Plan Composer

            THEN, execute every step you committed to, in order. After each tool
            call, briefly QUOTE that tool's output in your reasoning before moving
            to the next step. Gather data on at least 3 candidate recipes so the
            composer has options.

            Never repeat a tool call with identical arguments. Do NOT call
            compose_meal_plan yourself — that is the composer's job. Once your
            research is complete, hand off to the Meal Plan Composer with a brief
            summary of the candidates, their pantry percentages, and their costs.
            """,
            model=model,
            tools=[search_recipes, check_pantry, estimate_cost],
            handoffs=[composer],
        )


async def handle_runner_stream(runner: "Runner") -> List[str]:
    tool_calls_made: List[str] = []
    async for event in runner.stream_events():
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
            elif isinstance(event.data, ResponseOutputItemAddedEvent):
                if isinstance(event.data.item, ResponseFunctionToolCall):
                    tool_name = event.data.item.name
                    tool_calls_made.append(tool_name)
                    print(f"\n[Calling tool: {tool_name}]\n")
        elif event.type == "agent_updated_stream_event":
            new_agent_name = getattr(event.new_agent, "name", "?")
            print(f"\n[Handoff -> {new_agent_name}]\n", flush=True)
        elif event.type == "run_item_stream_event":
            if event.name == "handoff_requested":
                print("\n[Handoff requested]\n", flush=True)
            elif event.name == "handoff_occured" or event.name == "handoff_occurred":
                print("\n[Handoff completed]\n", flush=True)
            elif event.name == "tool_output" and isinstance(
                event.item, ToolCallOutputItem
            ):
                raw_item = event.item.raw_item
                content = (
                    raw_item.get("content")
                    if isinstance(raw_item, dict)
                    else getattr(raw_item, "content", "")
                )
                if content:
                    print(f"\n[Tool output: {str(content)[:200]}...]\n", end="", flush=True)
            elif event.name == "message_output_created":
                raw_item = event.item.raw_item
                role = getattr(raw_item, "role", None)
                if role is None and isinstance(raw_item, dict):
                    role = raw_item.get("role")
                if role == "assistant":
                    parts = []
                    for part in getattr(raw_item, "content", []):
                        if isinstance(part, ResponseOutputText):
                            parts.append(part.text)
                        elif isinstance(part, ResponseOutputRefusal):
                            parts.append(part.refusal)
                    if parts:
                        print("".join(parts), end="", flush=True)
    print()
    return tool_calls_made


async def run_recipe_query(query: str) -> List[str]:
    print("=" * 80)
    print(f"Query: {query}")
    print("=" * 80)
    ctx = RecipeContext(conversation_history=[])
    composer = MealPlanComposerAgent()
    researcher = RecipeResearcherAgent(composer=composer)
    print("\nAgent Response: ", end="", flush=True)
    messages = [{"role": "user", "content": query}]
    runner = Runner().run_streamed(starting_agent=researcher, input=messages)
    tool_calls = await handle_runner_stream(runner)
    print(f"\n{'=' * 80}")
    print(
        f"Query completed. Tools used: {', '.join(tool_calls) if tool_calls else 'None'}"
    )
    print(f"{'=' * 80}\n")
    return tool_calls


def generate_queries(n: int) -> List[str]:
    queries = [
        "Build me a 3-meal plan that is vegetarian and fits a $20 budget.",
        "Plan 3 vegan meals for under $20 total.",
        "Give me 3 vegetarian gluten-free meals on a $20 budget.",
        "I need a 3-meal vegetarian plan for the week, max $20.",
    ]
    return [random.choice(queries) for _ in range(n)]


async def main():
    parser = argparse.ArgumentParser(description="Recipe Planner Agent (efficient)")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()

    print("=" * 80)
    print("Recipe Planner Agent (efficient baseline) — OpenAI Agents SDK")
    print("=" * 80)
    print(f"Running {args.count} queries.")
    print("Expected efficiency: ~0.95+ (plan announced and fully executed)")
    print("=" * 80)

    queries = generate_queries(args.count)
    all_tool_calls = []
    for i, query in enumerate(queries, 1):
        print(f"\n\n{'#' * 80}\n# Query {i} of {args.count}\n{'#' * 80}\n")
        tool_calls = await run_recipe_query(query)
        all_tool_calls.append(
            {"query": query, "tools_used": tool_calls, "tool_count": len(tool_calls)}
        )
        if i < args.count:
            print(f"\nWaiting {args.delay}s...")
            time.sleep(args.delay)

    print("\n\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    tool_usage: Dict[str, int] = {}
    for r in all_tool_calls:
        for t in r["tools_used"]:
            tool_usage[t] = tool_usage.get(t, 0) + 1
    for tool, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {tool}: {count} times")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
