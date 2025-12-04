from typing import Dict, Any, Optional
from pydantic import BaseModel


class EvaluatorDetails(BaseModel):
    """
    Details for configuring an evaluator.

    Args:
        slug: The evaluator slug/identifier
        version: Optional version of the evaluator
        config: Optional configuration dictionary for the evaluator

    Example:
        >>> EvaluatorDetails(slug="pii-detector", config={"probability_threshold": 0.8})
        >>> EvaluatorDetails(slug="my-custom-evaluator", version="v2")
    """
    slug: str
    version: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
