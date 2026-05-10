#!/usr/bin/env python3
"""
Recipe Planner Handoff Demo for Traceloop.

Two specialized OpenAI Agents connected via a handoff:

  Recipe Researcher  → searches recipes, checks pantry, estimates cost,
                       then hands off to the composer.
  Meal Plan Composer → receives the researcher's findings and composes the
                       final 3-meal plan with the compose_meal_plan tool.

This exercises the handoff span path so you can verify the
``Recipe Researcher → Meal Plan Composer.handoff`` span (and the
``gen_ai.handoff.to_agent`` attribute) in Traceloop.
"""

import asyncio
import os
from typing import List

from agents import Agent, Runner, function_tool
from dotenv import load_dotenv
from pydantic import BaseModel
from traceloop.sdk import Traceloop
from traceloop.sdk.instruments import Instruments

load_dotenv()

Traceloop.init(
    app_name="recipe-planner-handoff-demo",
    disable_batch=True,
    instruments={Instruments.OPENAI, Instruments.OPENAI_AGENTS},
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


class CostEstimate(BaseModel):
    recipe_id: str
    cost_usd: float


class MealPlan(BaseModel):
    picks: List[str]
    total_cost_usd: float
    notes: str


@function_tool
async def search_recipes(constraints: List[str]) -> RecipeSearchResponse:
    """Return up to 5 recipes whose tags satisfy every constraint."""
    await asyncio.sleep(0.1)
    wanted = {c.lower() for c in constraints}
    matches = [
        Recipe(recipe_id=rid, name=r["name"], tags=r["tags"])
        for rid, r in RECIPE_DATABASE.items()
        if wanted.issubset({t.lower() for t in r["tags"]})
    ][:5]
    return RecipeSearchResponse(
        status="success",
        message=f"Found {len(matches)} recipes for {constraints}",
        recipes=matches,
    )


@function_tool
async def check_pantry(recipe_id: str) -> PantryCheck:
    """Return the fraction of a recipe's ingredients already on hand."""
    await asyncio.sleep(0.1)
    r = RECIPE_DATABASE[recipe_id]
    return PantryCheck(recipe_id=recipe_id, pantry_pct=r["pantry_pct"])


@function_tool
async def estimate_cost(recipe_id: str) -> CostEstimate:
    """Estimate the dollar cost to cook a recipe given current pantry levels."""
    await asyncio.sleep(0.1)
    r = RECIPE_DATABASE[recipe_id]
    return CostEstimate(recipe_id=recipe_id, cost_usd=r["cost_usd"])


@function_tool
async def compose_meal_plan(picks: List[str], notes: str = "") -> MealPlan:
    """Compose a final meal plan from selected recipe IDs."""
    await asyncio.sleep(0.1)
    total = sum(RECIPE_DATABASE[rid]["cost_usd"] for rid in picks if rid in RECIPE_DATABASE)
    return MealPlan(picks=picks, total_cost_usd=round(total, 2), notes=notes)


def build_agents() -> Agent:
    composer = Agent(
        name="Meal Plan Composer",
        handoff_description="Composes the final 3-meal plan from researched recipes.",
        instructions=(
            "You are the Meal Plan Composer. The Recipe Researcher has handed off to "
            "you with candidate recipes and their pantry/cost data. Pick exactly 3 "
            "that fit the user's constraints and budget, call compose_meal_plan once, "
            "then summarize the plan and confirm it fits the budget. Do NOT call any "
            "research tools — that work is already done."
        ),
        model="gpt-4o",
        tools=[compose_meal_plan],
    )

    researcher = Agent(
        name="Recipe Researcher",
        handoff_description="Searches recipes and gathers pantry + cost data.",
        instructions=(
            "You are the Recipe Researcher. Build the data the Meal Plan Composer "
            "needs to assemble a 3-meal plan within the user's dietary constraints "
            "and budget. Steps, in order: (1) search_recipes; (2) check_pantry for "
            "at least 3 candidates; (3) estimate_cost for those same candidates; "
            "(4) hand off to the Meal Plan Composer with a brief summary. Do NOT "
            "call compose_meal_plan yourself — that is the composer's job."
        ),
        model="gpt-4o",
        tools=[search_recipes, check_pantry, estimate_cost],
        handoffs=[composer],
    )

    return researcher


async def demo() -> None:
    researcher = build_agents()

    print("\n🚀 Starting recipe planner handoff workflow...")
    print("📊 Check Traceloop for the trace hierarchy.")

    query = "Build me a 3-meal plan that is vegetarian and fits a $20 budget."
    messages = [{"role": "user", "content": query}]
    runner = Runner().run_streamed(starting_agent=researcher, input=messages)

    async for event in runner.stream_events():
        if event.type == "agent_updated_stream_event":
            print(f"🔁 Active agent → {event.new_agent.name}")
        elif event.type == "run_item_stream_event":
            if "handoff" in event.name.lower():
                print(f"🔄 {event.name}")
            elif "tool" in event.name.lower():
                print(f"🔧 {event.name}")

    print("\n✅ Demo complete!")
    print("📊 Expected trace hierarchy in Traceloop:")
    print("   🌐 Agent Workflow")
    print("   ├─ 🤖 Recipe Researcher.agent")
    print("   │  ├─ 🔧 search_recipes.tool")
    print("   │  ├─ 🔧 check_pantry.tool (×N)")
    print("   │  └─ 🔧 estimate_cost.tool (×N)")
    print("   ├─ 🔄 Recipe Researcher → Meal Plan Composer.handoff")
    print("   └─ 🤖 Meal Plan Composer.agent")
    print("      └─ 🔧 compose_meal_plan.tool")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Set OPENAI_API_KEY environment variable")
        raise SystemExit(1)

    print("🎯 OpenAI Agents Handoff Demo — Recipe Planner")
    print("=" * 48)
    asyncio.run(demo())
