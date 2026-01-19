"""
Guardrail Example: Travel Agent with Safety Guards

This example demonstrates how to use the Traceloop SDK's guardrail feature
to protect LLM operations with runtime safety checks.

Examples shown:
1. Simple custom guard with lambda
2. Traceloop evaluator with Condition helpers
3. Agent trajectory evaluation
4. Custom on_failure handler with logging
"""

import asyncio
import json
import os

from openai import AsyncOpenAI

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, agent
from traceloop.sdk.guardrail import (
    GuardedOutput,
    Condition,
    OnFailure,
    GuardValidationError,
)
from traceloop.sdk.evaluator import EvaluatorMadeByTraceloop

# Initialize Traceloop FIRST - before any LLM calls
# Returns the client which provides access to guardrails
client = Traceloop.init(app_name="guardrail-travel-agent", disable_batch=True)

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Example 1: Simple Custom Guard with Lambda
# ==========================================
@workflow(name="simple_guard_example")
async def simple_guard_example():
    """Demonstrate a simple lambda-based guard for length validation."""

    async def generate_summary() -> GuardedOutput[str, dict]:
        """Generate a travel summary with length constraints."""
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Write a brief 2-sentence summary of Paris as a travel destination.",
                }
            ],
        )
        text = completion.choices[0].message.content
        return GuardedOutput(
            result=text,
            guard_input={"text": text, "word_count": len(text.split())},
        )

    try:
        result = await client.guardrails.run(
            func=generate_summary,
            guard=lambda z: z["word_count"] < 100,  # Max 100 words
            on_failure=OnFailure.raise_exception("Summary too long"),
        )
        print(f"Summary (passed guard): {result}")
    except GuardValidationError as e:
        print(f"Guard failed: {e}")


# Example 2: Traceloop Evaluator with Condition
# =============================================
@workflow(name="pii_guard_example")
async def pii_guard_example():
    """Demonstrate PII detection guard using Traceloop evaluator."""

    async def generate_customer_response() -> GuardedOutput[str, dict]:
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
        return GuardedOutput(
            result=text,
            guard_input={"text": text},  # PII detector expects 'text' field
        )

    result = await client.guardrails.run(
        func=generate_customer_response,
        guard=EvaluatorMadeByTraceloop.pii_detector(probability_threshold=0.7).as_guard(
            condition=Condition.is_false("has_pii")
        ),
        on_failure=OnFailure.log(message="PII detected in response"),
    )
    print(f"Customer response: {result[:100]}...")


# Example 3: Toxicity Guard with Score Threshold
# ==============================================
@workflow(name="toxicity_guard_example")
async def toxicity_guard_example():
    """Demonstrate toxicity detection with score-based condition."""

    async def generate_content() -> GuardedOutput[str, dict]:
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
        return GuardedOutput(
            result=text,
            guard_input={"text": text},
        )

    result = await client.guardrails.run(
        func=generate_content,
        guard=EvaluatorMadeByTraceloop.toxicity_detector(threshold=0.7).as_guard(
            condition=Condition.score_below(0.5, field="toxicity_score")
        ),
        on_failure=OnFailure.raise_exception("Content too toxic for family audience"),
    )
    print(f"Family-friendly content: {result[:100]}...")


# Example 4: Agent Trajectory Evaluation
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

    async def run_travel_agent() -> GuardedOutput[str, dict]:
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

        return GuardedOutput(
            result=final_response,
            guard_input=trajectory,
        )

    result = await client.guardrails.run(
        func=run_travel_agent,
        guard=EvaluatorMadeByTraceloop.agent_goal_completeness(threshold=0.7).as_guard(
            condition=Condition.score_above(0.6, field="completeness_score")
        ),
        on_failure=OnFailure.log(message="Agent did not fully complete the user's goal"),
    )
    print(f"Agent final response: {result[:100]}...")


# Example 5: Custom on_failure Handler with Alerting
# ==================================================
@workflow(name="custom_handler_example")
async def custom_handler_example():
    """Demonstrate custom on_failure handler with logging and alerting."""

    async def generate_travel_advice() -> GuardedOutput[str, dict]:
        """Generate travel advice that might need safety review."""
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "What should I know about traveling to a remote area?",
                }
            ],
        )
        text = completion.choices[0].message.content
        return GuardedOutput(
            result=text,
            guard_input={"text": text},
        )

    def custom_alert_handler(output: GuardedOutput) -> None:
        """Custom handler that logs and could send alerts."""
        print(f"[ALERT] Guard failed for output: {str(output.result)[:50]}...")
        print(f"[ALERT] Guard input was: {output.guard_input}")
        # In production, you might:
        # - Send to Slack/PagerDuty
        # - Log to monitoring system
        # - Store for review
        # For now, we raise to block the response
        raise GuardValidationError("Blocked after alerting team", output)

    try:
        result = await client.guardrails.run(
            func=generate_travel_advice,
            guard=lambda z: "danger" not in z["text"].lower(),  # Simple safety check
            on_failure=custom_alert_handler,
        )
        print(f"Travel advice: {result}")
    except GuardValidationError:
        print("Response was blocked by custom handler")


# Example 6: Shadow Mode (Evaluate but Don't Block)
# ================================================
@workflow(name="shadow_mode_example")
async def shadow_mode_example():
    """Demonstrate shadow mode for testing guards in production."""

    async def generate_response() -> GuardedOutput[str, dict]:
        """Generate a response to test with experimental evaluator."""
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Tell me about visa requirements for Europe."}],
        )
        text = completion.choices[0].message.content
        return GuardedOutput(
            result=text,
            guard_input={"text": text},
        )

    # Shadow mode: evaluate but don't block
    result = await client.guardrails.run(
        func=generate_response,
        guard=lambda z: len(z["text"]) > 50,  # Arbitrary check
        on_failure=OnFailure.noop(),  # Just observe, don't block
    )
    print(f"Response (shadow mode, always passes): {result[:100]}...")


# Example 7: Sync version with run_sync
# =====================================
def sync_guard_example():
    """Demonstrate synchronous guardrail usage."""

    def generate_simple_response() -> GuardedOutput[str, dict]:
        """Synchronous function that returns GuardedOutput."""
        text = "Welcome to our travel agency! How can I help you plan your trip?"
        return GuardedOutput(
            result=text,
            guard_input={"text": text, "length": len(text)},
        )

    result = client.guardrails.run_sync(
        func=generate_simple_response,
        guard=lambda z: z["length"] < 200,  # Max 200 characters
        on_failure=OnFailure.raise_exception("Response too long"),
    )
    print(f"Sync result: {result}")


async def main():
    """Run all guardrail examples."""
    print("=" * 60)
    print("Example 1: Simple Custom Guard with Lambda")
    print("=" * 60)
    await simple_guard_example()

    print("\n" + "=" * 60)
    print("Example 2: PII Detection Guard (requires Traceloop API key)")
    print("=" * 60)
    try:
        await pii_guard_example()
    except Exception as e:
        print(f"Skipped (API key required): {e}")

    print("\n" + "=" * 60)
    print("Example 3: Toxicity Guard (requires Traceloop API key)")
    print("=" * 60)
    try:
        await toxicity_guard_example()
    except Exception as e:
        print(f"Skipped (API key required): {e}")

    print("\n" + "=" * 60)
    print("Example 4: Agent Trajectory Evaluation (requires Traceloop API key)")
    print("=" * 60)
    try:
        await agent_trajectory_example()
    except Exception as e:
        print(f"Skipped (API key required): {e}")

    print("\n" + "=" * 60)
    print("Example 5: Custom on_failure Handler")
    print("=" * 60)
    await custom_handler_example()

    print("\n" + "=" * 60)
    print("Example 6: Shadow Mode (Evaluate but Don't Block)")
    print("=" * 60)
    await shadow_mode_example()

    print("\n" + "=" * 60)
    print("Example 7: Synchronous Guard")
    print("=" * 60)
    sync_guard_example()


if __name__ == "__main__":
    asyncio.run(main())
