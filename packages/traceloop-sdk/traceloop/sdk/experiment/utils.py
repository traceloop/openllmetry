"""
Shared utilities for running experiments with OpenTelemetry span capture.
"""

import json
from traceloop.sdk import Traceloop
from traceloop.sdk.utils.in_memory_span_exporter import InMemorySpanExporter
from traceloop.sdk.tracing.tracing import TracerWrapper


def extract_trajectory_from_spans(spans):
    """
    Extract prompt and completion trajectory from OpenTelemetry spans.
    Converts gen_ai.prompt.* to llm.prompts.* format expected by evaluators.

    Args:
        spans: List of ReadableSpan objects from InMemorySpanExporter

    Returns:
        dict with trajectory_prompts, trajectory_completions, and tool_calls
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

        for key, value in attributes.items():
            if key.startswith("gen_ai.prompt."):
                trajectory_prompts_dict[key] = value
            elif key.startswith("gen_ai.completion."):
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


async def run_with_span_capture(task_callable, *args, **kwargs):
    """
    Run a task with OpenTelemetry span capture and extract trajectory data.

    This function:
    1. Initializes Traceloop with InMemorySpanExporter
    2. Runs the provided async task callable
    3. Captures all OpenTelemetry spans
    4. Extracts prompt/completion trajectory from spans
    5. Returns trajectory data in JSON format

    Args:
        task_callable: Async callable to execute (e.g., run_travel_query)
        *args: Positional arguments to pass to the task callable
        **kwargs: Keyword arguments to pass to the task callable

    Returns:
        Tuple of (trajectory_prompts, trajectory_completions, final_completion)
        - trajectory_prompts: JSON string of prompt trajectory
        - trajectory_completions: JSON string of completion trajectory
        - final_completion: The final completion content string
    """
    # Clear singleton if existed to reinitialize with in-memory exporter
    if hasattr(TracerWrapper, "instance"):
        del TracerWrapper.instance

    # Create in-memory exporter to capture spans
    exporter = InMemorySpanExporter()

    # Initialize Traceloop with in-memory exporter
    Traceloop.init(
        app_name="internal-experiment-exporter",
        disable_batch=True,
        exporter=exporter,
    )

    try:
        # Run the task callable
        print(f"\n{'='*80}")
        print(f"Running task: {task_callable.__name__}")
        print(f"{'='*80}\n")

        tool_calls_made = await task_callable(*args, **kwargs)

        # Get all captured spans
        spans = exporter.get_finished_spans()

        print(f"\n{'='*80}")
        print(f"Captured {len(spans)} spans from execution")
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
        tools_called = ', '.join(trajectory_data['tool_calls']) if trajectory_data['tool_calls'] else 'None'
        print(f"  - Tools called: {tools_called}")
        if tool_calls_made:
            print(f"  - Tools from run: {', '.join(tool_calls_made) if tool_calls_made else 'None'}\n")

        json_trajectory_prompts = json.dumps(trajectory_prompts)
        json_trajectory_completions = json.dumps(trajectory_completions)

        return json_trajectory_prompts, json_trajectory_completions, final_completion

    except Exception as e:
        raise e
