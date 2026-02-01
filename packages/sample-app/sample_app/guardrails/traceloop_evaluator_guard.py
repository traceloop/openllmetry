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
    OnFailure,
    Guards,
)
from traceloop.sdk.generated.evaluators.request import (
    ToxicityDetectorInput,
    PIIDetectorInput,
    AgentGoalCompletenessInput,
)

# Initialize Traceloop - returns client with guardrails access
client = Traceloop.init(app_name="guardrail-traceloop-evaluator", disable_batch=True)

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Example 1: PII Detection Guard
# ==============================

async def generate_customer_response() -> str:
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
    return completion.choices[0].message.content or ""


@workflow(name="pii_guard_example")
async def pii_guard_example():
    """Demonstrate PII detection guard using Traceloop evaluator."""

    guardrail = client.guardrails.create(
        guards=[Guards.pii_detector(probability_threshold=0.7, timeout_in_sec=45)],
        on_failure=OnFailure.raise_exception(message="PII detected in response"),
    )
    result = await guardrail.run(
        generate_customer_response,
        input_mapper=lambda text: [PIIDetectorInput(text=text)],
    )
    print(f"Customer response: {result[:100]}...")


# Example 2: Toxicity Detection Guard
# ===================================
async def generate_content() -> str:
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
    return completion.choices[0].message.content or ""


@workflow(name="toxicity_guard_example")
async def toxicity_guard_example():
    """Demonstrate toxicity detection with score-based condition."""

    guardrail = client.guardrails.create(
        guards=[Guards.toxicity_detector(threshold=0.7)],
        on_failure=OnFailure.raise_exception("Content too toxic for family audience"),
    )
    result = await guardrail.run(
        generate_content,
        input_mapper=lambda text: [ToxicityDetectorInput(text=text)],
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
    response = completion.choices[0].message.content or ""
    state.completions.append(response)
    return response


@workflow(name="agent_trajectory_example")
async def agent_trajectory_example():
    """Demonstrate agent trajectory evaluation for goal completeness."""
    state = TravelAgentState()

    async def run_travel_agent() -> str:
        """Run the travel agent and return final response."""
        # Run the agent through multiple turns
        await travel_planner_agent(state, "I want to plan a 5-day trip to Japan.")
        final_response = await travel_planner_agent(
            state, "What are the must-see attractions in Tokyo?"
        )
        return final_response

    def create_trajectory_input(final_response: str) -> list:
        """Create trajectory input for evaluation."""
        # Format trajectory in the flattened dictionary format expected by the evaluator
        trajectory_prompts = {}
        for i, prompt in enumerate(state.prompts):
            trajectory_prompts[f"llm.prompts.{i}.role"] = "user"
            trajectory_prompts[f"llm.prompts.{i}.content"] = prompt

        trajectory_completions = {}
        for i, completion in enumerate(state.completions):
            trajectory_completions[f"llm.completions.{i}.content"] = completion

        return [AgentGoalCompletenessInput(
            trajectory_prompts=json.dumps(trajectory_prompts),
            trajectory_completions=json.dumps(trajectory_completions)
        )]

    guardrail = client.guardrails.create(
        guards=[Guards.agent_goal_completeness(threshold=0.7)],
        on_failure=OnFailure.return_value(value="Sorry the agent is unable to help you with that."),
    )
    result = await guardrail.run(run_travel_agent, input_mapper=create_trajectory_input)
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

    print("\n" + "=" * 60)
    print("Example 2: Toxicity Detection Guard")
    print("=" * 60)
    try:
        await toxicity_guard_example()
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Example 3: Agent Trajectory Evaluation")
    print("=" * 60)
    try:
        await agent_trajectory_example()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
