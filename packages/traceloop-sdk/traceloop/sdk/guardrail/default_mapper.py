"""
Default input mapper for common response types.

Handles automatic conversion of guarded function outputs to guard inputs.
"""
from typing import Any


def default_input_mapper(output: Any, num_guards: int) -> list[dict]:
    """
    Default mapper for common response types.

    Handles:
    - str: Creates dict with common text field names for each guard
    - dict with {question, answer, context}: Passes through with field aliases

    Args:
        output: The return value from the guarded function
        num_guards: Number of guards to create inputs for

    Returns:
        List of dicts, one per guard

    Raises:
        ValueError: If output type cannot be handled
    """
    if isinstance(output, str):
        # Map string to common field names used by evaluators
        input_dict = {
            "text": output,
            "prompt": output,
            "completion": output
        }
        return [input_dict] * num_guards

    if isinstance(output, dict):
        # Enrich dict with aliases for compatibility with various evaluators
        enriched = {**output}
        if "text" in output:
            enriched.setdefault("prompt", output["text"])
            enriched.setdefault("completion", output["text"])
        if "question" in output:
            enriched.setdefault("query", output["question"])
        if "answer" in output:
            enriched.setdefault("answer", output["answer"])
            enriched.setdefault("completion", output["answer"])
        if "context" in output:
            enriched.setdefault("context", [output["context"]])
        return [enriched] * num_guards

    raise ValueError(
        f"Default mapper cannot handle type {type(output).__name__}. "
        "Provide a custom input_mapper to run()."
    )
