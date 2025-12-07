from typing import Dict, Any, Optional, List
from pydantic import BaseModel


class EvaluatorDetails(BaseModel):
    """
    Details for configuring an evaluator.

    Args:
        slug: The evaluator slug/identifier
        version: Optional version of the evaluator
        config: Optional configuration dictionary for the evaluator
        required_input_fields: Optional list of required fields to the evaluator
            input. These fields must be present in the task output.

    Example:
        >>> EvaluatorDetails(slug="pii-detector", config={"probability_threshold": 0.8}, required_input_fields=["text"])
        >>> EvaluatorDetails(slug="my-custom-evaluator", version="v2")
    """
    slug: str
    version: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    required_input_fields: Optional[List[str]] = None
