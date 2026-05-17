from enum import Enum
from typing import Sequence
from traceloop.sdk.tracing.tracing import set_association_properties


class AssociationProperty(str, Enum):
    """Standard association properties for tracing."""

    CUSTOMER_ID = "customer_id"
    USER_ID = "user_id"
    SESSION_ID = "session_id"


# Type alias for a single association
Association = tuple[AssociationProperty, str]


class Associations:
    """Class for managing trace associations."""

    @staticmethod
    def set(associations: Sequence[Association]) -> None:
        """
        Set associations that will be added directly to all spans in the current context.

        Args:
            associations: A sequence of (property, value) tuples

        Example:
            # Single association
            traceloop.associations.set([(AssociationProperty.SESSION_ID, "conv-123")])

            # Multiple associations
            traceloop.associations.set([
                (AssociationProperty.USER_ID, "user-456"),
                (AssociationProperty.SESSION_ID, "session-789")
            ])
        """
        # Convert associations to a dict and use set_association_properties
        properties = {prop.value: value for prop, value in associations}
        set_association_properties(properties)
