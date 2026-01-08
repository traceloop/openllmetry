"""
Agent Evaluators Experiment

This example demonstrates Traceloop's agent tool trajectory evaluator:
- Agent Tool Trajectory: Validates the agent tool trajectory

This evaluator helps ensure your AI agents perform optimally and follow the expected tool trajectory.
"""

import asyncio
from traceloop.sdk import Traceloop
from traceloop.sdk.evaluator import EvaluatorMadeByTraceloop

# Initialize Traceloop
client = Traceloop.init()


async def agent_evaluators_task(row):
    executed_tool_calls = row.get("actual", "")
    expected_tool_calls = row.get(
        "expected",
        "[{'name': 'search', 'input': {'query': 'weather'}}, {'name': 'book_flight', 'input': {'flight': 'NYC to Paris'}}, {'name': 'get_confirmation', 'input': {'confirmation': 'flight booked'}}]"
    )

    return {
        "executed_tool_calls": executed_tool_calls,
        "expected_tool_calls": expected_tool_calls,
    }


async def run_agent_tool_trajectory_experiment():
    print("\n" + "="*80)
    print("AGENT TOOL TRAJECTORY EXPERIMENT")
    print("="*80 + "\n")
    print("This experiment will test the agent tool trajectory with the agent tool trajectory evaluator:\n")
    print("1. Agent Tool Trajectory - Validates the agent tool trajectory")
    print("\n" + "-"*80 + "\n")

    # Configure agent evaluators
    evaluators = [
        EvaluatorMadeByTraceloop.agent_tool_trajectory(
            input_params_sensitive=True,
            mismatch_sensitive=False,
            order_sensitive=False,
            threshold=0.7,
        ),
    ]

    print("Running experiment with evaluators:")
    for evaluator in evaluators:
        print(f"  - {evaluator.slug}")

    print("\n" + "-"*80 + "\n")

    # Run the experiment
    # Note: You'll need to create a dataset with appropriate test cases for agents
    results, errors = await client.experiment.run(
        dataset_slug="agent-tool-trajectory",  # Set a dataset slug that exists in the traceloop platform
        dataset_version="v1",
        task=agent_evaluators_task,
        evaluators=evaluators,
        experiment_slug="agent-tool-trajectory-exp",
        stop_on_error=False,
        wait_for_results=True,
    )

    print("\n" + "="*80)
    print("Agent tool trajectory experiment completed!")
    print("="*80 + "\n")

    print("Results summary:")
    print(f"  - Total rows processed: {len(results) if results else 0}")
    print(f"  - Errors encountered: {len(errors) if errors else 0}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")

if __name__ == "__main__":
    print("\nAgent Tool Trajectory Experiment\n")

    asyncio.run(run_agent_tool_trajectory_experiment())
