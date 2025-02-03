import pytest
from unittest.mock import Mock

# Assuming the set_span_attribute function is imported from the correct module
from opentelemetry.instrumentation.pinecone.utils import set_span_attribute

@pytest.mark.describe("Tests for set_span_attribute function")
class TestSetSpanAttribute:

    @pytest.mark.happy_path
    def test_set_attribute_with_valid_name_and_value(self):
        """
        Test that set_span_attribute sets the attribute when both name and value are valid.
        """
        span = Mock()
        set_span_attribute(span, "test_name", "test_value")
        span.set_attribute.assert_called_once_with("test_name", "test_value")

    @pytest.mark.happy_path
    def test_set_attribute_with_valid_name_and_empty_value(self):
        """
        Test that set_span_attribute does not set the attribute when the value is an empty string.
        """
        span = Mock()
        set_span_attribute(span, "test_name", "")
        span.set_attribute.assert_not_called()

    @pytest.mark.happy_path
    def test_set_attribute_with_valid_name_and_none_value(self):
        """
        Test that set_span_attribute does not set the attribute when the value is None.
        """
        span = Mock()
        set_span_attribute(span, "test_name", None)
        span.set_attribute.assert_not_called()

    @pytest.mark.edge_case
    def test_set_attribute_with_empty_name_and_valid_value(self):
        """
        Test that set_span_attribute sets the attribute when the name is empty but the value is valid.
        """
        span = Mock()
        set_span_attribute(span, "", "test_value")
        span.set_attribute.assert_called_once_with("", "test_value")

    @pytest.mark.edge_case
    def test_set_attribute_with_none_name_and_valid_value(self):
        """
        Test that set_span_attribute does not set the attribute when the name is None.
        """
        span = Mock()
        set_span_attribute(span, None, "test_value")
        span.set_attribute.assert_not_called()

    @pytest.mark.edge_case
    def test_set_attribute_with_special_characters_in_name_and_value(self):
        """
        Test that set_span_attribute sets the attribute when the name and value contain special characters.
        """
        span = Mock()
        set_span_attribute(span, "name!@#", "value$%^")
        span.set_attribute.assert_called_once_with("name!@#", "value$%^")

    @pytest.mark.edge_case
    def test_set_attribute_with_numeric_name_and_value(self):
        """
        Test that set_span_attribute sets the attribute when the name and value are numeric.
        """
        span = Mock()
        set_span_attribute(span, 123, 456)
        span.set_attribute.assert_called_once_with(123, 456)

# To run the tests, use the command: pytest -v