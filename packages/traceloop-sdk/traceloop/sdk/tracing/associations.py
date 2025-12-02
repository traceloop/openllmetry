from enum import Enum
from typing import Sequence
from opentelemetry import trace
from opentelemetry.context import attach, set_value, get_value


class AssociationProperty(str, Enum):
    """Standard association properties for tracing."""

    CONVERSATION_ID = "conversation_id"
    CUSTOMER_ID = "customer_id"
    USER_ID = "user_id"
    SESSION_ID = "session_id"


# Type alias for a single association
Association = tuple[AssociationProperty, str]


def set(associations: Sequence[Association]) -> None:
    """
    Set associations that will be added directly to all spans in the current context.

    Args:
        associations: A sequence of (property, value) tuples

    Example:
        # Single association
        Traceloop.associations.set([(AssociationProperty.CONVERSATION_ID, "conv-123")])

        # Multiple associations
        Traceloop.associations.set([
            (AssociationProperty.USER_ID, "user-456"),
            (AssociationProperty.SESSION_ID, "session-789")
        ])
    """
    # Store all associations in context
    current_associations = get_value("associations") or {}
    for prop, value in associations:
        if isinstance(prop, AssociationProperty):
            current_associations[prop.value] = value
        else:
            current_associations[str(prop)] = value

    attach(set_value("associations", current_associations))

    # Also set directly on the current span
    span = trace.get_current_span()
    if span and span.is_recording():
        for prop, value in associations:
            prop_name = prop.value if isinstance(prop, AssociationProperty) else str(prop)
            span.set_attribute(prop_name, value)
