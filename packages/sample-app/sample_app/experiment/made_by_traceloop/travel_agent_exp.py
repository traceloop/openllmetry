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
import importlib.util
from pathlib import Path

from traceloop.sdk import Traceloop
from traceloop.sdk.evaluator import EvaluatorMadeByTraceloop
from traceloop.sdk.utils.in_memory_span_exporter import InMemorySpanExporter
from traceloop.sdk.tracing.tracing import TracerWrapper

# Initialize Traceloop client (will be reinitialized per task with in-memory exporter)
client = Traceloop.init()

# Load travel_agent_example module dynamically
def _load_travel_agent_module():
    """Dynamically load the travel_agent_example module."""
    agents_path = Path(__file__).parent.parent.parent / "agents" / "travel_agent_example.py"
    spec = importlib.util.spec_from_file_location("travel_agent_example", agents_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["travel_agent_example"] = module
    spec.loader.exec_module(module)
    return module

# Load the module at import time
travel_agent_module = _load_travel_agent_module()


def extract_trajectory_from_spans(spans):
    """
    Extract prompt and completion trajectory from OpenTelemetry spans.

    Args:
        spans: List of ReadableSpan objects from InMemorySpanExporter

    Returns:
        dict with trajectory_prompts, trajectory_completions, tool_calls, tool_inputs, and tool_outputs
    """
    trajectory_prompts = []
    trajectory_completions = []
    tool_calls = []
    tool_inputs = []
    tool_outputs = []

    for span in spans:
        if not hasattr(span, 'attributes'):
            continue

        attributes = span.attributes or {}

        # Extract prompts (gen_ai.prompt.{i}.content)
        prompt_indices = set()
        for key in attributes.keys():
            if key.startswith("gen_ai.prompt.") and key.endswith(".content"):
                # Extract index from "gen_ai.prompt.{i}.content"
                parts = key.split(".")
                if len(parts) >= 3:
                    try:
                        idx = int(parts[2])
                        prompt_indices.add(idx)
                    except ValueError:
                        pass

        # Collect prompts in order
        for idx in sorted(prompt_indices):
            content_key = f"gen_ai.prompt.{idx}.content"
            if content_key in attributes:
                content = attributes[content_key]
                if content and content not in trajectory_prompts:
                    trajectory_prompts.append(content)

        # Extract completions (gen_ai.completion.{i}.content)
        completion_indices = set()
        for key in attributes.keys():
            if key.startswith("gen_ai.completion.") and key.endswith(".content"):
                # Extract index from "gen_ai.completion.{i}.content"
                parts = key.split(".")
                if len(parts) >= 3:
                    try:
                        idx = int(parts[2])
                        completion_indices.add(idx)
                    except ValueError:
                        pass

        # Collect completions in order
        for idx in sorted(completion_indices):
            content_key = f"gen_ai.completion.{idx}.content"
            if content_key in attributes:
                content = attributes[content_key]
                if content and content not in trajectory_completions:
                    trajectory_completions.append(content)

        # Extract tool calls and their inputs/outputs
        if "gen_ai.tool.name" in attributes:
            tool_name = attributes["gen_ai.tool.name"]
            if tool_name:
                tool_calls.append(tool_name)

                # Extract tool input (look for function call parameters)
                tool_input = attributes.get("gen_ai.completion.tool.arguments", "")
                if not tool_input:
                    # Try alternative attribute names
                    tool_input = attributes.get("gen_ai.tool.input", "")
                tool_inputs.append(tool_input)

                # Extract tool output (look for function results)
                tool_output = attributes.get("gen_ai.tool.output", "")
                if not tool_output:
                    # Try to find in span events or other attributes
                    tool_output = attributes.get("gen_ai.completion.tool.result", "")
                tool_outputs.append(tool_output)

    return {
        "trajectory_prompts": trajectory_prompts,
        "trajectory_completions": trajectory_completions,
        "tool_calls": tool_calls,
        "tool_inputs": tool_inputs,
        "tool_outputs": tool_outputs
    }


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
    # Use the dynamically loaded module
    run_travel_query = travel_agent_module.run_travel_query

    # Get query from row
    query = row.get("query", "Plan a 5-day trip to Paris")

    # Clear singleton if existed to reinitialize with in-memory exporter
    if hasattr(TracerWrapper, "instance"):
        _trace_wrapper_instance = TracerWrapper.instance
        del TracerWrapper.instance

    # Create in-memory exporter to capture spans
    exporter = InMemorySpanExporter()

    # Initialize Traceloop with in-memory exporter
    Traceloop.init(
        app_name="travel-agent-experiment",
        disable_batch=True,
        exporter=exporter,
    )

    try:
        # Run the travel agent query
        print(f"\n{'='*80}")
        print(f"Running travel agent for query: {query}")
        print(f"{'='*80}\n")

        tool_calls_made = await run_travel_query(query)

        # Get all captured spans
        spans = exporter.get_finished_spans()

        print(f"\n{'='*80}")
        print(f"Captured {len(spans)} spans from travel agent execution")
        print(f"{'='*80}\n")

        # Extract trajectory from spans
        trajectory_data = extract_trajectory_from_spans(spans)

        # Get the final completion (last completion in trajectory or empty string)
        final_completion = trajectory_data["trajectory_completions"][-1] if trajectory_data["trajectory_completions"] else ""

        # Combine all prompts and completions into strings for evaluator compatibility
        all_prompts = "\n\n---\n\n".join(trajectory_data["trajectory_prompts"])
        all_completions = "\n\n---\n\n".join(trajectory_data["trajectory_completions"])

        # Combine tool inputs and outputs for evaluators
        all_tool_inputs = "\n\n---\n\n".join(str(ti) for ti in trajectory_data["tool_inputs"]) if trajectory_data["tool_inputs"] else "No tool inputs captured"
        all_tool_outputs = "\n\n---\n\n".join(str(to) for to in trajectory_data["tool_outputs"]) if trajectory_data["tool_outputs"] else "No tool outputs captured"

        print(f"📊 Trajectory Summary:")
        print(f"  - Prompts captured: {len(trajectory_data['trajectory_prompts'])}")
        print(f"  - Completions captured: {len(trajectory_data['trajectory_completions'])}")
        print(f"  - Tools called: {', '.join(trajectory_data['tool_calls']) if trajectory_data['tool_calls'] else 'None'}")
        print(f"  - Tools from run: {', '.join(tool_calls_made) if tool_calls_made else 'None'}\n")

        # Return data using field synonyms that map to required evaluator fields
        # - "prompt" maps to "question"
        # - "answer" maps to "completion"
        # - "prompts" maps to "trajectory_prompts"
        # - "completions" maps to "trajectory_completions"
        # - "context" maps to "reference"
        # - "tool_input" and "tool_output" for agent_tool_error_detector
        return {
            "prompt": query, 
            "answer": final_completion,
            "context": f"The agent should create a complete travel itinerary for: {query}",
            "prompts": all_prompts,  
            "completions": all_completions, 
            "tool_input": all_tool_inputs,  # For agent_tool_error_detector
            "tool_output": all_tool_outputs,  # For agent_tool_error_detector
        }

    except Exception as e:
        print(f"\n❌ Error running travel agent: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            "prompt": query,
            "answer": f"Error: {str(e)}",
            "context": f"The agent should create a complete travel itinerary for: {query}",
            "prompts": "",
            "completions": "",
            "tool_input": "Error occurred before tool execution",
            "tool_output": f"Error: {str(e)}",
            "tool_calls": [],
            "tool_count": 0,
            "error": str(e)
        }
    finally:
        # Restore singleton if any
        if '_trace_wrapper_instance' in locals():
            TracerWrapper.instance = _trace_wrapper_instance


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

    if results:
        print("\nSample results:")
        for i, result in enumerate(results[:3], 1):
            print(f"\n  Result {i}:")
            if hasattr(result, 'task_result'):
                task_result = result.task_result
                print(f"    - Query: {task_result.get('prompt', 'N/A')[:100]}...")
                print(f"    - Tools used: {len(task_result.get('tool_calls', []))}")
                print(f"    - Prompts captured: {len(task_result.get('prompts', '').split('---'))}")
                print(f"    - Completions captured: {len(task_result.get('completions', '').split('---'))}")
            if hasattr(result, 'evaluations'):
                print(f"    - Evaluations: {list(result.evaluations.keys())}")


if __name__ == "__main__":
    print("\nTravel Agent Evaluators Experiment\n")
    print("This experiment captures the full prompt/completion trajectory")
    print("from the travel agent's execution and evaluates it against")
    print("Traceloop's agent evaluators.\n")

    asyncio.run(run_travel_agent_experiment())
