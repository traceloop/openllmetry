from unittest.mock import Mock

import pytest
# Import the function to be tested
from opentelemetry.instrumentation.anthropic.utils import set_span_attribute

# Describe block for set_span_attribute tests


@pytest.mark.describe("set_span_attribute function")
class TestSetSpanAttribute:

    @pytest.mark.happy_path
    def test_set_attribute_with_valid_value(self):
        """
        Test that set_span_attribute sets the attribute when a valid non-empty value is provided.
        """
        span = Mock()
        set_span_attribute(span, "test.attribute", "valid_value")
        span.set_attribute.assert_called_once_with("test.attribute", "valid_value")

    @pytest.mark.happy_path
    def test_set_attribute_with_none_value(self):
        """
        Test that set_span_attribute does not set the attribute when the value is None.
        """
        span = Mock()
        set_span_attribute(span, "test.attribute", None)
        span.set_attribute.assert_not_called()

    @pytest.mark.happy_path
    def test_set_attribute_with_empty_string(self):
        """
        Test that set_span_attribute does not set the attribute when the value is an empty string.
        """
        span = Mock()
        set_span_attribute(span, "test.attribute", "")
        span.set_attribute.assert_not_called()

    @pytest.mark.edge_case
    def test_set_attribute_with_whitespace_string(self):
        """
        Test that set_span_attribute sets the attribute when the value is a whitespace string.
        """
        span = Mock()
        set_span_attribute(span, "test.attribute", "   ")
        span.set_attribute.assert_called_once_with("test.attribute", "   ")

    @pytest.mark.edge_case
    def test_set_attribute_with_special_characters(self):
        """
        Test that set_span_attribute sets the attribute when the value contains special characters.
        """
        span = Mock()
        special_value = "!@#$%^&*()_+"
        set_span_attribute(span, "test.attribute", special_value)
        span.set_attribute.assert_called_once_with("test.attribute", special_value)

    @pytest.mark.edge_case
    def test_set_attribute_with_numeric_value(self):
        """
        Test that set_span_attribute sets the attribute when the value is a numeric type.
        """
        span = Mock()
        numeric_value = 12345
        set_span_attribute(span, "test.attribute", numeric_value)
        span.set_attribute.assert_called_once_with("test.attribute", numeric_value)

    @pytest.mark.edge_case
    def test_set_attribute_with_boolean_value(self):
        """
        Test that set_span_attribute sets the attribute when the value is a boolean type.
        """
        span = Mock()
        boolean_value = True
        set_span_attribute(span, "test.attribute", boolean_value)
        span.set_attribute.assert_called_once_with("test.attribute", boolean_value)


# Run the tests
if __name__ == "__main__":
    pytest.main()
