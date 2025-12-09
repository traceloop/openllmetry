"""
Travel Agent Evaluators Experiment

This example demonstrates running the travel agent and collecting prompt/completion
trajectory through the run for evaluation with Traceloop's agent evaluators:
- Agent Goal Accuracy: Validates agent goal achievement
- Agent Tool Error Detector: Detects errors or failures during tool execution
- Agent Flow Quality: Validates agent trajectories against user-defined natural language tests
- Agent Efficiency: Evaluates agent efficiency by checking for redundant calls and optimal paths
- Agent Goal Completeness: Measures whether the agent successfully accomplished all user goals

The key feature is extracting the full prompt/completion trajectory from OpenTelemetry spans
for detailed analysis and evaluation.
"""

import asyncio
import sys
from pathlib import Path

from traceloop.sdk import Traceloop
from traceloop.sdk.evaluator import EvaluatorMadeByTraceloop
from traceloop.sdk.experiment.utils import run_with_span_capture

# Add the agents directory to sys.path for imports
agents_dir = Path(__file__).parent.parent.parent / "agents"
if str(agents_dir) not in sys.path:
    sys.path.insert(0, str(agents_dir))

from travel_agent_example import run_travel_query  # noqa: E402

# Initialize Traceloop client (will be reinitialized per task with in-memory exporter)
client = Traceloop.init()


async def travel_agent_task(row):
    """
    Unified task function for travel agent evaluators.

    This task:
    1. Initializes Traceloop with InMemorySpanExporter
    2. Runs the travel agent with the query from the dataset
    3. Captures all OpenTelemetry spans
    4. Extracts prompt/completion trajectory from spans
    5. Returns data in format compatible with agent evaluators

    Required fields for agent evaluators:
    - question (or prompt): The input question or goal
    - completion (or answer, response, text): The agent's final response
    - trajectory_prompts (or prompts): The agent's prompt trajectory
    - trajectory_completions (or completions): The agent's completion trajectory
    - tool_calls: List of tools called during execution
    """
    # Get query from row
    query = row.get("query", "Plan a 5-day trip to Paris")

    # Run the travel agent with span capture
    trajectory_prompts, trajectory_completions, final_completion = await run_with_span_capture(
        run_travel_query,  # This is the function that calls the Agent
        query  # This is the agents input
    )

    return {
        "prompt": query,
        "answer": final_completion if final_completion else query,
        "context": f"The agent should create a complete travel itinerary for: {query}",
        "trajectory_prompts": trajectory_prompts,
        "trajectory_completions": trajectory_completions,
    }


async def run_travel_agent_experiment():
    """
    Run experiment with travel agent and all 5 agent evaluators.

    This experiment will evaluate the travel agent's performance across:
    1. Agent Goal Accuracy - Did the agent achieve the stated goal?
    2. Agent Tool Error Detector - Were there any tool execution errors?
    3. Agent Flow Quality - Did the agent follow the expected trajectory?
    4. Agent Efficiency - Was the agent efficient (no redundant calls)?
    5. Agent Goal Completeness - Did the agent fully accomplish all goals?
    """

    print("\n" + "="*80)
    print("TRAVEL AGENT EVALUATORS EXPERIMENT")
    print("="*80 + "\n")

    print("This experiment will test the travel agent with five agent-specific evaluators:\n")
    print("1. Agent Goal Accuracy - Validates goal achievement")
    print("2. Agent Tool Error Detector - Detects tool execution errors")
    print("3. Agent Flow Quality - Validates expected trajectories")
    print("4. Agent Efficiency - Checks for optimal execution paths")
    print("5. Agent Goal Completeness - Measures full goal accomplishment")
    print("\n" + "-"*80 + "\n")

    # Configure agent evaluators
    evaluators = [
        EvaluatorMadeByTraceloop.agent_goal_accuracy(),
        EvaluatorMadeByTraceloop.agent_flow_quality(
            threshold=0.7,
            conditions=["create_itinerary tool should be called last"],
        ),
        EvaluatorMadeByTraceloop.agent_efficiency(),
        EvaluatorMadeByTraceloop.agent_goal_completeness(),
    ]

    print("Running experiment with evaluators:")
    for evaluator in evaluators:
        print(f"  - {evaluator.slug}")

    print("\n" + "-"*80 + "\n")

    # Run the experiment
    # Note: You'll need to create a dataset with travel queries in the Traceloop platform
    results, errors = await client.experiment.run(
        dataset_slug="travel-queries",  # Dataset slug that should exist in traceloop platform
        dataset_version="v1",
        task=travel_agent_task,
        evaluators=evaluators,
        experiment_slug="travel-agent-exp",
        stop_on_error=False,
        wait_for_results=True,
    )

    print("\n" + "="*80)
    print("Travel agent evaluators experiment completed!")
    print("="*80 + "\n")

    print("Results summary:")
    print(f"  - Total rows processed: {len(results) if results else 0}")
    print(f"  - Errors encountered: {len(errors) if errors else 0}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")

if __name__ == "__main__":
    print("\nTravel Agent Evaluators Experiment\n")
    print("This experiment captures the full prompt/completion trajectory")
    print("from the travel agent's execution and evaluates it against")
    print("Traceloop's agent evaluators.\n")

    asyncio.run(run_travel_agent_experiment())
