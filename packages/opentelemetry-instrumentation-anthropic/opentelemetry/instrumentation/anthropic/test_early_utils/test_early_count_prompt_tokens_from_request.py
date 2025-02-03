import pytest
from unittest.mock import Mock

# Assuming the function is imported from the correct module
from opentelemetry.instrumentation.anthropic.utils import count_prompt_tokens_from_request

@pytest.mark.describe("Tests for count_prompt_tokens_from_request")
class TestCountPromptTokensFromRequest:

    @pytest.mark.happy_path
    def test_single_prompt_string(self):
        """
        Test with a single prompt string to ensure the function counts tokens correctly.
        """
        anthropic = Mock()
        anthropic.count_tokens = Mock(return_value=5)
        request = {"prompt": "Hello, world!"}
        
        result = count_prompt_tokens_from_request(anthropic, request)
        
        assert result == 5
        anthropic.count_tokens.assert_called_once_with("Hello, world!")

    @pytest.mark.happy_path
    def test_multiple_messages_with_string_content(self):
        """
        Test with multiple messages containing string content to ensure correct token counting.
        """
        anthropic = Mock()
        anthropic.count_tokens = Mock(side_effect=[3, 4])
        request = {
            "messages": [
                {"content": "Hi"},
                {"content": "How are you?"}
            ]
        }
        
        result = count_prompt_tokens_from_request(anthropic, request)
        
        assert result == 7
        anthropic.count_tokens.assert_any_call("Hi")
        anthropic.count_tokens.assert_any_call("How are you?")

    @pytest.mark.happy_path
    def test_messages_with_list_content(self):
        """
        Test with messages containing list content to ensure correct token counting.
        """
        anthropic = Mock()
        anthropic.count_tokens = Mock(side_effect=[2, 3])
        request = {
            "messages": [
                {"content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]}
            ]
        }
        
        result = count_prompt_tokens_from_request(anthropic, request)
        
        assert result == 5
        anthropic.count_tokens.assert_any_call("Hello")
        anthropic.count_tokens.assert_any_call("World")

    @pytest.mark.edge_case
    def test_empty_request(self):
        """
        Test with an empty request to ensure the function returns zero tokens.
        """
        anthropic = Mock()
        request = {}
        
        result = count_prompt_tokens_from_request(anthropic, request)
        
        assert result == 0
        anthropic.count_tokens.assert_not_called()

    @pytest.mark.edge_case
    def test_no_count_tokens_method(self):
        """
        Test when the anthropic object does not have a count_tokens method.
        """
        anthropic = Mock()
        del anthropic.count_tokens
        request = {"prompt": "Hello, world!"}
        
        result = count_prompt_tokens_from_request(anthropic, request)
        
        assert result == 0

    @pytest.mark.edge_case
    def test_non_string_content_in_messages(self):
        """
        Test with non-string content in messages to ensure they are ignored.
        """
        anthropic = Mock()
        anthropic.count_tokens = Mock(return_value=0)
        request = {
            "messages": [
                {"content": 123},
                {"content": {"type": "image", "url": "http://example.com/image.png"}}
            ]
        }
        
        result = count_prompt_tokens_from_request(anthropic, request)
        
        assert result == 0
        anthropic.count_tokens.assert_not_called()

    @pytest.mark.edge_case
    def test_mixed_content_types_in_list(self):
        """
        Test with mixed content types in a list to ensure only text types are counted.
        """
        anthropic = Mock()
        anthropic.count_tokens = Mock(return_value=3)
        request = {
            "messages": [
                {"content": [{"type": "text", "text": "Hello"}, {"type": "image", "url": "http://example.com/image.png"}]}
            ]
        }
        
        result = count_prompt_tokens_from_request(anthropic, request)
        
        assert result == 3
        anthropic.count_tokens.assert_called_once_with("Hello")