"""
Factory class for creating Traceloop evaluators with proper configuration.

This module dynamically generates factory methods from the generated.evaluators registry.
"""

from typing import Any, List

from ..generated.evaluators import REQUEST_MODELS
from .config import EvaluatorDetails


def _get_required_fields(slug: str) -> List[str]:
    """Get required input fields for an evaluator from its request model."""
    model = REQUEST_MODELS.get(slug)
    if not model:
        return []

    input_field = model.model_fields.get("input")
    if not input_field or not input_field.annotation:
        return []

    input_model = input_field.annotation
    if not hasattr(input_model, "model_fields"):
        return []

    return [
        name for name, field in input_model.model_fields.items()
        if field.is_required()
    ]


def _get_config_fields(slug: str) -> dict:
    """Get config fields with their defaults from the request model."""
    model = REQUEST_MODELS.get(slug)
    if not model:
        return {}

    config_field = model.model_fields.get("config")
    if not config_field or not config_field.annotation:
        return {}

    config_annotation = config_field.annotation
    origin = getattr(config_annotation, "__origin__", None)
    if origin is not None:
        args = getattr(config_annotation, "__args__", ())
        config_model = next((a for a in args if a is not type(None)), None)
    else:
        config_model = config_annotation

    if not config_model or not hasattr(config_model, "model_fields"):
        return {}

    config_fields = {}
    for name, field in config_model.model_fields.items():
        config_fields[name] = field.default
    return config_fields


def _slug_to_method_name(slug: str) -> str:
    """Convert slug like 'pii-detector' to method name like 'pii_detector'."""
    return slug.replace("-", "_")


def _method_name_to_slug(method_name: str) -> str:
    """Convert method name like 'pii_detector' to slug like 'pii-detector'."""
    return method_name.replace("_", "-")


def create_evaluator(slug: str, **config: Any) -> EvaluatorDetails:
    """Create an EvaluatorDetails for the given slug with optional config.

    Args:
        slug: The evaluator slug (e.g., "pii-detector")
        **config: Configuration options for the evaluator

    Returns:
        EvaluatorDetails configured for the specified evaluator

    Example:
        >>> from traceloop.sdk.evaluator import create_evaluator
        >>> evaluator = create_evaluator("pii-detector", probability_threshold=0.8)
    """
    if slug not in REQUEST_MODELS:
        available = ", ".join(sorted(REQUEST_MODELS.keys()))
        raise ValueError(f"Unknown evaluator slug: '{slug}'. Available: {available}")

    # Remove None values from config
    config = {k: v for k, v in config.items() if v is not None}
    return EvaluatorDetails(
        slug=slug,
        version=None,
        config=config,
        required_input_fields=_get_required_fields(slug),
    )


class _EvaluatorMadeByTraceloopMeta(type):
    """Metaclass that dynamically generates evaluator factory methods."""

    def __getattr__(cls, name: str) -> Any:
        """Dynamically create factory methods for any evaluator slug."""
        slug = _method_name_to_slug(name)
        if slug in REQUEST_MODELS:

            def factory(**config: Any) -> EvaluatorDetails:
                return create_evaluator(slug, **config)

            factory.__name__ = name
            config_fields = list(_get_config_fields(slug).keys()) or "none"
            factory.__doc__ = f"Create {slug} evaluator. Config fields: {config_fields}"
            return factory
        raise AttributeError(f"'{cls.__name__}' has no attribute '{name}'")

    def __dir__(cls) -> List[str]:
        """List all available evaluator methods."""
        methods = list(super().__dir__())
        for slug in REQUEST_MODELS:
            methods.append(_slug_to_method_name(slug))
        return methods


class EvaluatorMadeByTraceloop(metaclass=_EvaluatorMadeByTraceloopMeta):
    """
    Factory class for creating Traceloop evaluators with proper configuration.

    All evaluator slugs from the registry are available as methods.
    Methods are dynamically generated from REQUEST_MODELS.

    Example:
        >>> from traceloop.sdk.evaluator import EvaluatorMadeByTraceloop
        >>>
        >>> evaluators = [
        ...     EvaluatorMadeByTraceloop.pii_detector(probability_threshold=0.8),
        ...     EvaluatorMadeByTraceloop.toxicity_detector(threshold=0.7),
        ...     EvaluatorMadeByTraceloop.faithfulness(),
        ... ]

    Available evaluators (auto-generated from registry):
        - pii_detector, toxicity_detector, prompt_injection
        - regex_validator, json_validator, sql_validator
        - faithfulness, answer_relevancy, context_relevance
        - agent_goal_accuracy, agent_efficiency, agent_flow_quality
        - and more... (use dir(EvaluatorMadeByTraceloop) to see all)
    """

    pass
