import pytest
from unittest.mock import Mock

# Assuming the set_span_attribute function is imported from the correct module
from opentelemetry.instrumentation.haystack.utils import set_span_attribute

@pytest.mark.describe("set_span_attribute")
class TestSetSpanAttribute:
    
    @pytest.mark.happy_path
    def test_set_span_attribute_with_valid_value(self):
        """
        Test that set_span_attribute sets the attribute on the span when a valid value is provided.
        """
        span = Mock()
        name = "test.attribute"
        value = "test_value"
        
        set_span_attribute(span, name, value)
        
        span.set_attribute.assert_called_once_with(name, value)

    @pytest.mark.happy_path
    def test_set_span_attribute_with_empty_string(self):
        """
        Test that set_span_attribute does not set the attribute when the value is an empty string.
        """
        span = Mock()
        name = "test.attribute"
        value = ""
        
        set_span_attribute(span, name, value)
        
        span.set_attribute.assert_not_called()

    @pytest.mark.happy_path
    def test_set_span_attribute_with_none_value(self):
        """
        Test that set_span_attribute does not set the attribute when the value is None.
        """
        span = Mock()
        name = "test.attribute"
        value = None
        
        set_span_attribute(span, name, value)
        
        span.set_attribute.assert_not_called()

    @pytest.mark.edge_case
    def test_set_span_attribute_with_numeric_value(self):
        """
        Test that set_span_attribute sets the attribute on the span when a numeric value is provided.
        """
        span = Mock()
        name = "test.attribute"
        value = 123
        
        set_span_attribute(span, name, value)
        
        span.set_attribute.assert_called_once_with(name, value)

    @pytest.mark.edge_case
    def test_set_span_attribute_with_boolean_value(self):
        """
        Test that set_span_attribute sets the attribute on the span when a boolean value is provided.
        """
        span = Mock()
        name = "test.attribute"
        value = True
        
        set_span_attribute(span, name, value)
        
        span.set_attribute.assert_called_once_with(name, value)

    @pytest.mark.edge_case
    def test_set_span_attribute_with_special_characters(self):
        """
        Test that set_span_attribute sets the attribute on the span when the value contains special characters.
        """
        span = Mock()
        name = "test.attribute"
        value = "!@#$%^&*()_+"
        
        set_span_attribute(span, name, value)
        
        span.set_attribute.assert_called_once_with(name, value)

    @pytest.mark.edge_case
    def test_set_span_attribute_with_large_string(self):
        """
        Test that set_span_attribute sets the attribute on the span when a very large string is provided.
        """
        span = Mock()
        name = "test.attribute"
        value = "a" * 10000  # Large string
        
        set_span_attribute(span, name, value)
        
        span.set_attribute.assert_called_once_with(name, value)

# To run the tests, use the command: pytest -v