import os
import pytest
from unittest import mock
from opentelemetry.instrumentation.haystack.utils import should_send_prompts
from opentelemetry import context as context_api

@pytest.mark.describe("Tests for should_send_prompts function")
class TestShouldSendPrompts:

    @pytest.mark.happy_path
    def test_should_send_prompts_env_var_true(self):
        """Test should_send_prompts returns True when TRACELOOP_TRACE_CONTENT is 'true'."""
        with mock.patch.dict(os.environ, {"TRACELOOP_TRACE_CONTENT": "true"}):
            assert should_send_prompts() is True

   
    @pytest.mark.happy_path
    def test_should_send_prompts_env_var_not_set(self):
        """Test should_send_prompts returns True when TRACELOOP_TRACE_CONTENT is not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            assert should_send_prompts() is True

    

    @pytest.mark.edge_case
    def test_should_send_prompts_env_var_case_insensitivity(self):
        """Test should_send_prompts handles TRACELOOP_TRACE_CONTENT case insensitively."""
        with mock.patch.dict(os.environ, {"TRACELOOP_TRACE_CONTENT": "TrUe"}):
            assert should_send_prompts() is True

    
    @pytest.mark.edge_case
    def test_should_send_prompts_no_env_var_and_no_context(self):
        """Test should_send_prompts returns True when neither TRACELOOP_TRACE_CONTENT nor context is set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            context_api.set_value("override_enable_content_tracing", None)
            assert should_send_prompts() is True

# To run the tests, use the command: pytest -v