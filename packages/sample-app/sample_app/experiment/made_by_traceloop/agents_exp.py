"""
Agent Evaluators Experiment

This example demonstrates Traceloop's agent evaluators:
- Agent Goal Accuracy: Validates agent goal achievement
- Agent Tool Error Detector: Detects errors or failures during tool execution
- Agent Flow Quality: Validates agent trajectories against user-defined natural language tests
- Agent Efficiency: Evaluates agent efficiency by checking for redundant calls and optimal paths
- Agent Goal Completeness: Measures whether the agent successfully accomplished all user goals

These evaluators help ensure your AI agents perform optimally and achieve their objectives.
"""

import asyncio
import os
from openai import AsyncOpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.evaluator import EvaluatorMadeByTraceloop

# Initialize Traceloop
client = Traceloop.init()


async def generate_agent_trace(task_description: str) -> dict:
    """
    Simulate an agent execution and generate trace data.
    In a real scenario, this would come from your actual agent framework.
    """
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Simulate agent executing a task
    response = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful agent that completes tasks step by step."},
            {"role": "user", "content": task_description}
        ],
        temperature=0.7,
        max_tokens=300,
    )

    completion = response.choices[0].message.content

    # Return trace data (simplified for demo)
    # In production, this would be actual trace/span data from your agent
    return {
        "task": task_description,
        "completion": completion,
    }


async def agent_evaluators_task(row):
    """
    Unified task function for all 5 agent evaluators.

    IMPORTANT: Thanks to field synonym mapping, you can use flexible field names!
    For example:
    - "answer", "response", "text" → all map to "completion"
    - "prompt", "instructions" → map to "question"
    - "context", "ground_truth" → map to "reference"
    - "prompts" → maps to "trajectory_prompts"
    - "completions" → maps to "trajectory_completions"

    This makes it easier to write tasks without worrying about exact field names.

    Required fields for the 5 agent evaluators:
    - question (or prompt): The input question or goal (for agent_goal_accuracy)
    - completion (or answer, response, text): The agent's completion (for agent_goal_accuracy)
    - reference (or ground_truth, context): The reference answer (for agent_goal_accuracy)
    - tool_input: The input to tools (for agent_tool_error_detector)
    - tool_output: The output from tools (for agent_tool_error_detector)
    - trajectory_prompts (or prompts): The agent's prompt trajectory
        (for agent_flow_quality, agent_efficiency, agent_goal_completeness)
    - trajectory_completions (or completions): The agent's completion trajectory
        (for agent_flow_quality, agent_efficiency, agent_goal_completeness)
    """
    # Get data from row or use defaults
    question = row.get("question", "Book a flight from New York to Paris")
    reference = row.get(
        "reference",
        "Successfully booked flight NYC to Paris, departure 2024-12-15, return 2024-12-22"
    )
    tool_input = row.get("tool_input", "New York to Paris")
    tool_output = row.get(
        "tool_output",
        "Successfully booked flight NYC to Paris, departure 2024-12-15, return 2024-12-22"
    )
    trajectory_prompts = row.get("trajectory_prompts", "New York to Paris")
    trajectory_completions = row.get(
        "trajectory_completions",
        "Successfully booked flight NYC to Paris, departure 2024-12-15, return 2024-12-22"
    )

    # Generate agent trace
    trace_data = await generate_agent_trace(question)

    # You can use synonyms! These will automatically map to the required fields:
    # - Using "answer" instead of "completion" ✓
    # - Using "prompt" instead of "question" ✓
    # - Using "context" instead of "reference" ✓
    return {
        "prompt": question,  # Maps to "question"
        "answer": trace_data["completion"],  # Maps to "completion"
        "context": reference,  # Maps to "reference"
        "tool_input": tool_input,
        "tool_output": tool_output,
        "prompts": trajectory_prompts,  # Maps to "trajectory_prompts"
        "completions": trajectory_completions,  # Maps to "trajectory_completions"
    }


async def run_agents_experiment():
    """
    Run experiment with all 5 agent evaluators.

    This experiment will evaluate agent performance across:
    1. Agent Goal Accuracy - Did the agent achieve the stated goal?
    2. Agent Tool Error Detector - Were there any tool execution errors?
    3. Agent Flow Quality - Did the agent follow the expected trajectory?
    4. Agent Efficiency - Was the agent efficient (no redundant calls)?
    5. Agent Goal Completeness - Did the agent fully accomplish all goals?
    """

    print("\n" + "="*80)
    print("AGENT EVALUATORS EXPERIMENT")
    print("="*80 + "\n")

    print("This experiment will test five agent-specific evaluators:\n")
    print("1. Agent Goal Accuracy - Validates goal achievement")
    print("2. Agent Tool Error Detector - Detects tool execution errors")
    print("3. Agent Flow Quality - Validates expected trajectories")
    print("4. Agent Efficiency - Checks for optimal execution paths")
    print("5. Agent Goal Completeness - Measures full goal accomplishment")
    print("\n" + "-"*80 + "\n")

    # Configure agent evaluators
    evaluators = [
        EvaluatorMadeByTraceloop.agent_goal_accuracy(),
        EvaluatorMadeByTraceloop.agent_tool_error_detector(),
        EvaluatorMadeByTraceloop.agent_flow_quality(),
        EvaluatorMadeByTraceloop.agent_efficiency(),
        EvaluatorMadeByTraceloop.agent_goal_completeness(),
    ]

    print("Running experiment with evaluators:")
    for evaluator in evaluators:
        print(f"  - {evaluator.slug}")

    print("\n" + "-"*80 + "\n")

    # Run the experiment
    # Note: You'll need to create a dataset with appropriate test cases for agents
    results, errors = await client.experiment.run(
        dataset_slug="agents",  # Set a dataset slug that exists in the traceloop platform
        dataset_version="v1",
        task=agent_evaluators_task,
        evaluators=evaluators,
        experiment_slug="agents-evaluators-exp",
        stop_on_error=False,
        wait_for_results=True,
    )

    print("\n" + "="*80)
    print("Agent evaluators experiment completed!")
    print("="*80 + "\n")

    print("Results summary:")
    print(f"  - Total rows processed: {len(results) if results else 0}")
    print(f"  - Errors encountered: {len(errors) if errors else 0}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")

if __name__ == "__main__":
    print("\nAgent Evaluators Experiment\n")

    asyncio.run(run_agents_experiment())
