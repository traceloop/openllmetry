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
import json
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
    Converts gen_ai.prompt.* to llm.prompts.* format expected by evaluators.

    Args:
        spans: List of ReadableSpan objects from InMemorySpanExporter

    Returns:
        dict with trajectory_prompts (as dict with llm.prompts.* keys), trajectory_completions, and tool_calls
    """
    # Collect all gen_ai attributes and convert to llm.prompts/completions format
    trajectory_prompts_dict = {}
    trajectory_completions_dict = {}
    tool_calls = []
    tool_inputs = []
    tool_outputs = []

    for span in spans:
        if not hasattr(span, 'attributes'):
            continue

        attributes = span.attributes or {}

        # Convert gen_ai.prompt.* to llm.prompts.* format
        for key, value in attributes.items():
            if key.startswith("gen_ai.prompt."):
                # Convert gen_ai.prompt.X.Y to llm.prompts.X.Y
                new_key = key.replace("gen_ai.prompt.", "llm.prompts.")
                trajectory_prompts_dict[new_key] = value
            elif key.startswith("gen_ai.completion."):
                # Convert gen_ai.completion.X.Y to llm.completions.X.Y
                new_key = key.replace("gen_ai.completion.", "llm.completions.")
                trajectory_completions_dict[new_key] = value
            # Also check for existing llm.* attributes
            elif key.startswith("llm.prompts."):
                trajectory_prompts_dict[key] = value
            elif key.startswith("llm.completions."):
                trajectory_completions_dict[key] = value

        # Extract tool calls for summary
        if "gen_ai.tool.name" in attributes:
            tool_name = attributes["gen_ai.tool.name"]
            if tool_name:
                tool_calls.append(tool_name)

                # Extract tool input
                tool_input = attributes.get("gen_ai.completion.tool.arguments", "")
                if not tool_input:
                    tool_input = attributes.get("gen_ai.tool.input", "")
                tool_inputs.append(tool_input)

                # Extract tool output
                tool_output = attributes.get("gen_ai.tool.output", "")
                if not tool_output:
                    tool_output = attributes.get("gen_ai.completion.tool.result", "")
                tool_outputs.append(tool_output)

    return {
        "trajectory_prompts": trajectory_prompts_dict,
        "trajectory_completions": trajectory_completions_dict,
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

        # Get the final completion from llm.completions dict
        completions_dict = trajectory_data["trajectory_completions"]
        final_completion = ""
        if completions_dict:
            # Find the highest index completion content
            max_idx = -1
            for key in completions_dict.keys():
                if ".content" in key:
                    try:
                        parts = key.split(".")
                        idx = int(parts[2])
                        if idx > max_idx:
                            max_idx = idx
                            final_completion = completions_dict[key]
                    except (ValueError, IndexError):
                        pass

        # trajectory_prompts and trajectory_completions are dicts with llm.prompts/completions.* keys
        # If empty, use JSON string fallback to avoid validation errors
        trajectory_prompts = trajectory_data["trajectory_prompts"]
        trajectory_completions = trajectory_data["trajectory_completions"]

        # Convert to JSON strings if empty (evaluators expect string when no data)
        if not trajectory_prompts:
            trajectory_prompts = json.dumps([])
        if not trajectory_completions:
            trajectory_completions = json.dumps([])

        print("📊 Trajectory Summary:")
        print(f"  - Prompt attributes captured: {len(trajectory_prompts)}")
        print(f"  - Completion attributes captured: {len(trajectory_completions)}")
        print(f"  - Tools called: {', '.join(trajectory_data['tool_calls']) if trajectory_data['tool_calls'] else 'None'}")
        print(f"  - Tools from run: {', '.join(tool_calls_made) if tool_calls_made else 'None'}\n")

        # Return data using field synonyms that map to required evaluator fields
        # - "prompt" maps to "question"
        # - "answer" maps to "completion"
        # - "context" maps to "reference"
        # - "trajectory_prompts" and "trajectory_completions" as dicts with llm.prompts/completions.* keys
        
        json_trajectory_prompts = json.dumps(trajectory_prompts)
        json_trajectory_completions = json.dumps(trajectory_completions)
        # prompt_list = str(trajectory_prompts)
        # completion_list = str(trajectory_completions)

        return {
            "prompt": query,
            "answer": final_completion if final_completion else query,
            "context": f"The agent should create a complete travel itinerary for: {query}",
            "trajectory_prompts": json_trajectory_prompts,  
            "trajectory_completions": json_trajectory_completions,  
        }

    except Exception as e:
        raise e


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
        experiment_slug="travel-agent-exp-2",
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
