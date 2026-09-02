from collections.abc import Hashable
from unittest.mock import MagicMock

import pytest

from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
    ChatStream,
    _sanitize_attributes_for_metrics,
)
from opentelemetry.instrumentation.openai.shared.config import Config


class TestUnhashableAttributesFix:
    def test_sanitize_attributes_with_lists(self):
        attributes_with_lists = {
            "model": "gpt-4",
            "tools_required": ["param1", "param2"],
            "tool_params": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
            "simple_string": "test",
            "simple_number": 42,
        }

        sanitized = _sanitize_attributes_for_metrics(attributes_with_lists)

        for key, value in sanitized.items():
            assert isinstance(value, Hashable)
            hash(value)

        assert sanitized["model"] == "gpt-4"
        assert sanitized["simple_string"] == "test"
        assert sanitized["simple_number"] == 42
        assert sanitized["tools_required"] == '["param1", "param2"]'
        assert '"location"' in sanitized["tool_params"]

    def test_sanitize_attributes_with_none_values(self):
        attributes = {
            "none_value": None,
            "empty_list": [],
            "empty_dict": {},
            "normal_value": "test",
        }

        sanitized = _sanitize_attributes_for_metrics(attributes)
        assert sanitized == {
            "none_value": None,
            "empty_list": "[]",
            "empty_dict": "{}",
            "normal_value": "test",
        }

    def test_sanitize_attributes_preserves_hashable_values(self):
        hashable_attributes = {
            "string": "test",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "tuple": (1, 2, 3),
        }

        sanitized = _sanitize_attributes_for_metrics(hashable_attributes)
        assert sanitized == hashable_attributes

    def test_config_get_common_metrics_attributes_with_unhashable_values(
        self, monkeypatch
    ):
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
                                "required": ["location"]
                            }
                        }
                    }
                ],
                "other_config": {"nested": {"data": "value"}},
            }

        monkeypatch.setattr(Config, "get_common_metrics_attributes", mock_get_common_attributes)
        chat_stream = ChatStream(
            span=MagicMock(),
            response=MagicMock(),
            instance=MagicMock(),
            start_time=1234567890.0,
            request_kwargs={"model": "gpt-4"},
        )

        attributes = chat_stream._shared_attributes()
        for value in attributes.values():
            assert isinstance(value, Hashable)
            hash(value)

    def test_original_issue_reproduction_simulation(self):
        problematic_attributes = {
            "gen_ai.system": "openai",
            "gen_ai.response.model": "gpt-4",
            "gen_ai.operation.name": "chat",
            "server.address": "https://api.openai.com/v1",
            "stream": True,
            "tool_required_params": ["location", "unit"],
            "tool_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }

        try:
            frozenset(problematic_attributes.items())
            pytest.fail("Expected TypeError was not raised - test setup may be incorrect")
        except TypeError:
            pass

        sanitized = _sanitize_attributes_for_metrics(problematic_attributes)

        try:
            frozenset(sanitized.items())
        except TypeError as e:
            pytest.fail(f"Sanitization failed to fix the issue: {e}")

        assert sanitized["gen_ai.system"] == "openai"
        assert sanitized["gen_ai.response.model"] == "gpt-4"
        assert sanitized["stream"] is True
