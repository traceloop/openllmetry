import os
import pytest
from unittest.mock import patch
from opentelemetry.instrumentation.anthropic.utils import should_send_prompts

@pytest.mark.describe("Tests for should_send_prompts function")
class TestShouldSendPrompts:

    @pytest.mark.happy_path
    def test_should_send_prompts_env_var_true(self):
        """
        Test that should_send_prompts returns True when the TRACELOOP_TRACE_CONTENT
        environment variable is set to 'true'.
        """
        with patch.dict(os.environ, {"TRACELOOP_TRACE_CONTENT": "true"}):
            assert should_send_prompts() is True

    @pytest.mark.happy_path
    def test_should_send_prompts_env_var_true_case_insensitive(self):
        """
        Test that should_send_prompts returns True when the TRACELOOP_TRACE_CONTENT
        environment variable is set to 'TRUE' (case insensitive).
        """
        with patch.dict(os.environ, {"TRACELOOP_TRACE_CONTENT": "TRUE"}):
            assert should_send_prompts() is True


    @pytest.mark.happy_path
    def test_should_send_prompts_env_var_not_set(self):
        """
        Test that should_send_prompts returns True when the TRACELOOP_TRACE_CONTENT
        environment variable is not set, defaulting to 'true'.
        """
        with patch.dict(os.environ, {}, clear=True):
            assert should_send_prompts() is True

    @pytest.mark.edge_case
    def test_should_send_prompts_override_enable_content_tracing(self):
        """
        Test that should_send_prompts returns True when the context API has
        'override_enable_content_tracing' set to True, regardless of the environment variable.
        """
        with patch.dict(os.environ, {"TRACELOOP_TRACE_CONTENT": "false"}):
            with patch('opentelemetry.context.get_value', return_value=True):
                assert should_send_prompts() is True

    @pytest.mark.edge_case
    def test_should_send_prompts_override_enable_content_tracing_false(self):
        """
        Test that should_send_prompts returns False when the context API has
        'override_enable_content_tracing' set to False and the environment variable is 'false'.
        """
        with patch.dict(os.environ, {"TRACELOOP_TRACE_CONTENT": "false"}):
            with patch('opentelemetry.context.get_value', return_value=False):
                assert should_send_prompts() is False

    @pytest.mark.edge_case
    def test_should_send_prompts_no_env_var_no_override(self):
        """
        Test that should_send_prompts returns True when neither the environment variable
        nor the context API override is set, defaulting to 'true'.
        """
        with patch.dict(os.environ, {}, clear=True):
            with patch('opentelemetry.context.get_value', return_value=False):
                assert should_send_prompts() is True