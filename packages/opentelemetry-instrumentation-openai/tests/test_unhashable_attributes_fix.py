"""
Test to reproduce and verify the fix for the unhashable attributes issue #3428.

This test simulates the scenario where OpenAI tool definitions with lists
cause TypeError when streaming chat completions are used.
"""
import pytest
from unittest.mock import MagicMock, patch
from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
    _sanitize_attributes_for_metrics,
    ChatStream,
)
from opentelemetry.instrumentation.openai.shared.config import Config


class TestUnhashableAttributesFix:
    def test_sanitize_attributes_with_lists(self):
        """Test that _sanitize_attributes_for_metrics handles unhashable values"""
        # Test attributes with unhashable types (lists, dicts)
        attributes_with_lists = {
            "model": "gpt-4",
            "tools_required": ["param1", "param2"],  # This list causes the issue
            "tool_params": {"type": "object", "properties": {"location": {"type": "string"}}},  # Dict
            "simple_string": "test",
            "simple_number": 42,
        }
        
        # This should not raise an exception
        sanitized = _sanitize_attributes_for_metrics(attributes_with_lists)
        
        # Verify all values are now hashable
        for key, value in sanitized.items():
            try:
                # Test if the value is hashable by creating a frozenset
                frozenset({key: value}.items())
            except TypeError:
                pytest.fail(f"Value for key '{key}' is still not hashable: {value}")
        
        # Verify the content is preserved (as JSON strings for complex types)
        assert sanitized["model"] == "gpt-4"
        assert sanitized["simple_string"] == "test"
        assert sanitized["simple_number"] == 42
        assert '"param1"' in sanitized["tools_required"]  # Should be JSON string
        assert '"location"' in sanitized["tool_params"]   # Should be JSON string

    def test_sanitize_attributes_with_none_values(self):
        """Test that None values and empty values are handled correctly"""
        attributes = {
            "none_value": None,
            "empty_list": [],
            "empty_dict": {},
            "normal_value": "test"
        }
        
        sanitized = _sanitize_attributes_for_metrics(attributes)
        
        # Verify all are hashable
        for key, value in sanitized.items():
            try:
                frozenset({key: value}.items())
            except TypeError:
                pytest.fail(f"Value for key '{key}' is still not hashable: {value}")

    def test_sanitize_attributes_preserves_hashable_values(self):
        """Test that already hashable values are preserved as-is"""
        hashable_attributes = {
            "string": "test",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "tuple": (1, 2, 3),  # Tuples are hashable
        }
        
        sanitized = _sanitize_attributes_for_metrics(hashable_attributes)
        
        # All values should remain the same
        assert sanitized == hashable_attributes

    def test_config_get_common_metrics_attributes_with_unhashable_values(self):
        """Test that the fix works when Config.get_common_metrics_attributes returns unhashable values"""
        # Mock Config.get_common_metrics_attributes to return unhashable data
        original_fn = Config.get_common_metrics_attributes
        
        def mock_get_common_attributes():
            return {
                "tool_definitions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"]  # This list causes the crash
                            }
                        }
                    }
                ],
                "other_config": {"nested": {"data": "value"}}
            }
        
        try:
            Config.get_common_metrics_attributes = mock_get_common_attributes
            
            # Create a mock ChatStream instance
            mock_span = MagicMock()
            mock_response = MagicMock()
            mock_instance = MagicMock()
            
            # Create ChatStream with mocked objects
            chat_stream = ChatStream(
                span=mock_span,
                response=mock_response,
                instance=mock_instance,
                start_time=1234567890.0,
                request_kwargs={"model": "gpt-4"}
            )
            
            # This should not raise TypeError: unhashable type: 'list'
            attributes = chat_stream._shared_attributes()
            
            # Verify all attributes are hashable
            for key, value in attributes.items():
                try:
                    frozenset({key: value}.items())
                except TypeError:
                    pytest.fail(f"Attribute '{key}' with value '{value}' is not hashable")
                    
        finally:
            # Restore original function
            Config.get_common_metrics_attributes = original_fn

    def test_original_issue_reproduction_simulation(self):
        """
        Simulate the original issue scenario from the bug report.
        This test reproduces the conditions that led to the TypeError.
        """
        # Simulate the problematic attributes that would come from OpenAI tool definitions
        problematic_attributes = {
            "gen_ai.system": "openai",
            "gen_ai.response.model": "gpt-4",
            "gen_ai.operation.name": "chat",
            "server.address": "https://api.openai.com/v1",
            "stream": True,
            # This simulates what happens when tool definitions leak into attributes
            "tool_required_params": ["location", "unit"],  # This causes the crash
            "tool_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]  # Another list that causes issues
            }
        }
        
        # Before the fix, this would fail when trying to create a frozenset
        try:
            frozenset(problematic_attributes.items())
            pytest.fail("Expected TypeError was not raised - test setup may be incorrect")
        except TypeError:
            # This is expected - the original issue
            pass
        
        # After applying our sanitization, it should work
        sanitized = _sanitize_attributes_for_metrics(problematic_attributes)
        
        # This should not raise an exception
        try:
            frozenset(sanitized.items())
        except TypeError as e:
            pytest.fail(f"Sanitization failed to fix the issue: {e}")
        
        # Verify important attributes are preserved
        assert sanitized["gen_ai.system"] == "openai"
        assert sanitized["gen_ai.response.model"] == "gpt-4"
        assert sanitized["stream"] is True