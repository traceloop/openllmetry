"""
Guardrail Example: Using Traceloop Evaluators as Guards

This example demonstrates how to use Traceloop's built-in evaluators
(PII detection, toxicity, agent goal completeness) as guards.

Requires a Traceloop API key to run the evaluators.
"""

import asyncio
import json
import os

from openai import AsyncOpenAI

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, agent
from traceloop.sdk.guardrail import (
    GuardedFunctionOutput,
    Condition,
    OnFailure,
)
from traceloop.sdk.generated.evaluators.request import ToxicityDetectorInput, PIIDetectorInput
from traceloop.sdk.evaluator import EvaluatorMadeByTraceloop

# Initialize Traceloop - returns client with guardrails access
client = Traceloop.init(app_name="guardrail-traceloop-evaluator", disable_batch=True)

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Example 1: PII Detection Guard
# ==============================
@workflow(name="pii_guard_example")
async def pii_guard_example():
    """Demonstrate PII detection guard using Traceloop evaluator."""

    async def generate_customer_response() -> GuardedFunctionOutput[str, dict]:
        """Generate a customer service response."""
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful travel agent. Never include personal information.",
                },
                {
                    "role": "user",
                    "content": "What are the best beaches in Thailand?",
                },
            ],
        )
        text = completion.choices[0].message.content
        return GuardedFunctionOutput(
            result=text,
            guard_input=PIIDetectorInput(text=text),
        )

    result = await client.guardrails.run(
        func_to_guard=generate_customer_response,
        guard=EvaluatorMadeByTraceloop.pii_detector(
            probability_threshold=0.7
        ).as_guard(condition=Condition.is_false(field="has_pii"), timeout_in_sec=45),
        on_failure=OnFailure.log(message="PII detected in response"),
    )
    print(f"Customer response: {result[:100]}...")


# Example 2: Toxicity Detection Guard
# ===================================
@workflow(name="toxicity_guard_example")
async def toxicity_guard_example():
    """Demonstrate toxicity detection with score-based condition."""

    async def generate_content() -> GuardedFunctionOutput[str, dict]:
        """Generate travel content that should be family-friendly."""
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Write a family-friendly description of nightlife in Tokyo.",
                }
            ],
        )
        text = completion.choices[0].message.content
        return GuardedFunctionOutput(
            result=text,
            guard_input=ToxicityDetectorInput(text=text),
        )

    result = await client.guardrails.run(
        func_to_guard=generate_content,
        guard=EvaluatorMadeByTraceloop.toxicity_detector(threshold=0.7).as_guard(
            condition=Condition.score_below(0.5, field="toxicity_score")
        ),
        on_failure=OnFailure.raise_exception("Content too toxic for family audience"),
    )
    print(f"Family-friendly content: {result[:100]}...")


# Example 3: Agent Trajectory Evaluation
# =====================================
class TravelAgentState:
    """Track agent prompts and completions for trajectory evaluation."""

    def __init__(self):
        self.prompts = []
        self.completions = []


@agent(name="travel_planner")
async def travel_planner_agent(state: TravelAgentState, query: str) -> str:
    """A travel planning agent that tracks its trajectory."""
    state.prompts.append(query)

    completion = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert travel planner. Provide helpful travel advice.",
            },
            {"role": "user", "content": query},
        ],
    )
    response = completion.choices[0].message.content
    state.completions.append(response)
    return response


@workflow(name="agent_trajectory_example")
async def agent_trajectory_example():
    """Demonstrate agent trajectory evaluation for goal completeness."""
    state = TravelAgentState()

    async def run_travel_agent() -> GuardedFunctionOutput[str, dict]:
        """Run the travel agent and prepare trajectory for evaluation."""
        # Run the agent through multiple turns
        await travel_planner_agent(state, "I want to plan a 5-day trip to Japan.")
        final_response = await travel_planner_agent(
            state, "What are the must-see attractions in Tokyo?"
        )

        # Prepare trajectory for evaluation
        trajectory = {
            "trajectory_prompts": json.dumps(state.prompts),
            "trajectory_completions": json.dumps(state.completions),
        }

        return GuardedFunctionOutput(
            result=final_response,
            guard_input=trajectory,
        )

    result = await client.guardrails.run(
        func_to_guard=run_travel_agent,
        guard=EvaluatorMadeByTraceloop.agent_goal_completeness(threshold=0.7).as_guard(
            condition=Condition.score_above(0.6, field="completeness_score")
        ),
        on_failure=OnFailure.log(message="Agent did not fully complete the user's goal"),
    )
    print(f"Agent final response: {result[:100]}...")


async def main():
    """Run all Traceloop evaluator guard examples."""
    print("=" * 60)
    print("Example 1: PII Detection Guard")
    print("=" * 60)
    try:
        await pii_guard_example()
    except Exception as e:
        print(f"Error: {e}")

    # print("\n" + "=" * 60)
    # print("Example 2: Toxicity Detection Guard")
    # print("=" * 60)
    # try:
    #     await toxicity_guard_example()
    # except Exception as e:
    #     print(f"Error: {e}")

    # print("\n" + "=" * 60)
    # print("Example 3: Agent Trajectory Evaluation")
    # print("=" * 60)
    # try:
    #     await agent_trajectory_example()
    # except Exception as e:
    #     print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
