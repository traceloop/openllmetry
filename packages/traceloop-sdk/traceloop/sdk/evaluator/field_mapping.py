"""
Field mapping utilities for evaluator input normalization.

This module provides synonym mapping to allow users to use flexible field names
in their task outputs, which are automatically normalized to match evaluator requirements.
"""

from typing import Dict, Any, Set, List


# Synonym groups - each group contains fields that can be used interchangeably
# The first field in each group is the "canonical" form that gets preserved
SYNONYM_GROUPS = [
    # Output/Response fields - the most common synonyms
    {"text", "completion", "answer", "response"},

    # Reference/Truth/Context fields - source of truth for comparison
    {"reference", "ground_truth", "context"},

    # Input/Question fields - user input variations
    {"question", "prompt", "instructions", "query"},

    # Conversation/Trajectory prompts
    {"prompts", "trajectory_prompts"},

    # Conversation/Trajectory completions
    {"completions", "trajectory_completions"},
]


def _build_synonym_map() -> Dict[str, Set[str]]:
    """
    Build a mapping from each field to all its synonyms (including itself).

    Returns:
        Dict mapping each field name to the set of all its synonyms
    """
    synonym_map: Dict[str, Set[str]] = {}

    for group in SYNONYM_GROUPS:
        # Each field in the group maps to all fields in the group
        for field in group:
            synonym_map[field] = group

    return synonym_map


# Pre-built synonym map for efficient lookup
_SYNONYM_MAP = _build_synonym_map()


def get_synonyms(field: str) -> Set[str]:
    """
    Get all synonyms for a given field name.

    Args:
        field: The field name to get synonyms for

    Returns:
        Set of all synonym field names (including the field itself)
        If field has no synonyms, returns a set containing only the field
    """
    return _SYNONYM_MAP.get(field, {field})


def normalize_task_output(
    task_output: Dict[str, Any],
    required_fields: List[str],
) -> Dict[str, Any]:
    """
    Normalize task output field names to match required evaluator fields.

    This function maps user-provided field names to the evaluator's required field names
    using synonym groups. Original fields that are mapped to required fields are removed,
    while fields that don't participate in mapping are preserved.

    For example, if a user returns {"answer": "..."} but the evaluator needs "completion",
    this function will create {"completion": "..."} and remove "answer".

    Args:
        task_output: The dictionary returned by the task function
        required_fields: List of field names required by the evaluator

    Returns:
        A new dictionary with normalized field names that match required_fields.
        Original synonym fields are removed, but unmapped fields are preserved.

    Example:
        >>> task_output = {"answer": "Paris", "prompt": "What is the capital?"}
        >>> required = ["completion", "question"]
        >>> normalize_task_output(task_output, required)
        {"completion": "Paris", "question": "What is the capital?"}
    """
    normalized = {}
    mapped_keys: Set[str] = set()

    # First pass: map required fields from task output
    for required_field in required_fields:
        # If the exact required field already exists, use it (prioritize exact match)
        if required_field in task_output:
            normalized[required_field] = task_output[required_field]
            mapped_keys.add(required_field)
            continue

        # Get all possible synonyms for this required field
        synonyms = get_synonyms(required_field)

        # Find which synonym (if any) exists in the task output
        found_key = None
        for synonym in synonyms:
            if synonym in task_output:
                found_key = synonym
                break

        if found_key:
            # Map the found field to the required field name
            normalized[required_field] = task_output[found_key]
            mapped_keys.add(found_key)

    # Second pass: preserve fields that weren't mapped
    # (they might be needed by other evaluators or for debugging)
    for key, value in task_output.items():
        if key not in mapped_keys and key not in normalized:
            normalized[key] = value

    return normalized


def get_field_suggestions(missing_field: str, available_fields: List[str]) -> List[str]:
    """
    Get suggestions for a missing field based on available fields and synonyms.

    Args:
        missing_field: The required field that is missing
        available_fields: List of field names available in task output

    Returns:
        List of suggested field names that could satisfy the requirement
    """
    suggestions = []
    synonyms = get_synonyms(missing_field)

    # Check if any available field is a synonym of the missing field
    for available in available_fields:
        available_synonyms = get_synonyms(available)
        if synonyms & available_synonyms:  # If there's any overlap
            suggestions.append(available)

    return suggestions


def format_field_help(field: str) -> str:
    """
    Format help text showing all accepted synonyms for a field.

    Args:
        field: The field name to get help for

    Returns:
        Formatted string showing the field and its synonyms
    """
    synonyms = get_synonyms(field)
    if len(synonyms) == 1:
        return f"'{field}'"

    synonym_list = sorted(synonyms)
    return f"'{field}' (or synonyms: {', '.join(repr(s) for s in synonym_list if s != field)})"
