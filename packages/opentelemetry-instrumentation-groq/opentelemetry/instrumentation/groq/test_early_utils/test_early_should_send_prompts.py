import os
import pytest
from unittest import mock
from opentelemetry.instrumentation.groq.utils import should_send_prompts

# Describe block for should_send_prompts tests
@pytest.mark.describe("Tests for should_send_prompts function")
class TestShouldSendPrompts:

    @pytest.mark.happy_path
    def test_should_send_prompts_env_var_true(self):
        """
        Test that should_send_prompts returns True when TRACELOOP_TRACE_CONTENT is set to 'true'.
        """
        with mock.patch.dict(os.environ, {"TRACELOOP_TRACE_CONTENT": "true"}):
            assert should_send_prompts() is True

    @pytest.mark.happy_path
    def test_should_send_prompts_env_var_true_case_insensitive(self):
        """
        Test that should_send_prompts returns True when TRACELOOP_TRACE_CONTENT is set to 'TRUE' (case insensitive).
        """
        with mock.patch.dict(os.environ, {"TRACELOOP_TRACE_CONTENT": "TRUE"}):
            assert should_send_prompts() is True

    @pytest.mark.happy_path
    def test_should_send_prompts_env_var_not_set(self):
        """
        Test that should_send_prompts returns True when TRACELOOP_TRACE_CONTENT is not set and context_api.get_value returns True.
        """
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch('opentelemetry.context.get_value', return_value=True):
                assert should_send_prompts() is True

    @pytest.mark.edge_case
    def test_should_send_prompts_context_override(self):
        """
        Test that should_send_prompts returns True when context_api.get_value returns True, regardless of TRACELOOP_TRACE_CONTENT.
        """
        with mock.patch.dict(os.environ, {"TRACELOOP_TRACE_CONTENT": "false"}):
            with mock.patch('opentelemetry.context.get_value', return_value=True):
                assert should_send_prompts() is True



# Run the tests
if __name__ == "__main__":
    pytest.main()