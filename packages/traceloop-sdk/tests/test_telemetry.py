import os
import sys
from unittest import mock
import pytest
from traceloop.sdk import Traceloop
from traceloop.sdk.telemetry import Telemetry


class TestTelemetry:
    def setup_method(self):
        # Clear any existing Telemetry singleton instance before each test
        if hasattr(Telemetry, 'instance'):
            delattr(Telemetry, 'instance')
        
        # Reset global telemetry state
        Telemetry._global_telemetry_enabled = None
        
        # Clear environment variable
        if 'TRACELOOP_TELEMETRY' in os.environ:
            del os.environ['TRACELOOP_TELEMETRY']

    def teardown_method(self):
        # Clean up after each test
        if hasattr(Telemetry, 'instance'):
            delattr(Telemetry, 'instance')
        
        # Reset global telemetry state
        Telemetry._global_telemetry_enabled = None
        
        # Clear environment variable
        if 'TRACELOOP_TELEMETRY' in os.environ:
            del os.environ['TRACELOOP_TELEMETRY']

    def test_telemetry_disabled_via_init_parameter(self):
        """Test that telemetry_enabled=False in Traceloop.init() should disable telemetry"""
        # Mock PostHog to track if it gets initialized
        with mock.patch('traceloop.sdk.telemetry.Posthog') as mock_posthog:
            # Initialize Traceloop with telemetry explicitly disabled
            Traceloop.init(
                app_name="test_app",
                api_endpoint="http://localhost:8080",  # Use non-traceloop endpoint to avoid API key requirement
                telemetry_enabled=False,
                exporter=mock.Mock()  # Mock exporter to avoid endpoint issues
            )
            
            # Verify that PostHog was not initialized (telemetry should be disabled)
            mock_posthog.assert_not_called()
            
            # Create a Telemetry instance and verify it's disabled
            telemetry = Telemetry()
            assert not telemetry._telemetry_enabled, "Telemetry should be disabled when telemetry_enabled=False"

    def test_telemetry_disabled_in_pytest(self):
        """Test that telemetry is disabled when running in pytest"""
        # Mock PostHog to avoid actual network calls
        with mock.patch('traceloop.sdk.telemetry.Posthog') as mock_posthog:
            # Initialize Traceloop with default telemetry settings
            Traceloop.init(
                app_name="test_app", 
                api_endpoint="http://localhost:8080",  # Use non-traceloop endpoint to avoid API key requirement
                exporter=mock.Mock()  # Mock exporter to avoid endpoint issues
            )
            
            # Verify that PostHog was NOT initialized (telemetry should be disabled in pytest)  
            mock_posthog.assert_not_called()
            
            # Create a Telemetry instance and verify it's disabled
            telemetry = Telemetry()
            assert not telemetry._telemetry_enabled, "Telemetry should be disabled when running in pytest"

    def test_telemetry_disabled_via_env_var(self):
        """Test that TRACELOOP_TELEMETRY=false disables telemetry"""
        # Set environment variable to disable telemetry
        os.environ['TRACELOOP_TELEMETRY'] = 'false'
        
        # Mock PostHog to track if it gets initialized
        with mock.patch('traceloop.sdk.telemetry.Posthog') as mock_posthog:
            # Initialize Traceloop
            Traceloop.init(
                app_name="test_app",
                api_endpoint="http://localhost:8080",  # Use non-traceloop endpoint to avoid API key requirement
                exporter=mock.Mock()  # Mock exporter to avoid endpoint issues
            )
            
            # Verify that PostHog was not initialized
            mock_posthog.assert_not_called()
            
            # Create a Telemetry instance and verify it's disabled
            telemetry = Telemetry()
            assert not telemetry._telemetry_enabled, "Telemetry should be disabled when TRACELOOP_TELEMETRY=false"

    def test_init_parameter_overrides_env_var(self):
        """Test that telemetry_enabled parameter should override environment variable"""
        # Set environment variable to enable telemetry
        os.environ['TRACELOOP_TELEMETRY'] = 'true'
        
        # Mock PostHog to track if it gets initialized
        with mock.patch('traceloop.sdk.telemetry.Posthog') as mock_posthog:
            # Initialize Traceloop with telemetry explicitly disabled via parameter
            Traceloop.init(
                app_name="test_app",
                api_endpoint="http://localhost:8080",  # Use non-traceloop endpoint to avoid API key requirement
                telemetry_enabled=False,  # This should override the env var
                exporter=mock.Mock()  # Mock exporter to avoid endpoint issues
            )
            
            # Verify that PostHog was not initialized (parameter should override env var)
            mock_posthog.assert_not_called()
            
            # This test will currently fail because the bug exists
            # The Telemetry singleton only checks the environment variable, not the init parameter

    def test_telemetry_bug_reproduction_real_scenario(self):
        """Test the actual scenario described in GitHub issue #3175"""
        # Clear env var first to ensure clean state
        if 'TRACELOOP_TELEMETRY' in os.environ:
            del os.environ['TRACELOOP_TELEMETRY']
            
        # Mock PostHog to track calls
        with mock.patch('traceloop.sdk.telemetry.Posthog') as mock_posthog:
            mock_posthog_instance = mock.Mock()
            mock_posthog.return_value = mock_posthog_instance
            
            # This is the actual sequence from the bug report:
            # 1. Initialize with telemetry_enabled=False
            Traceloop.init(
                app_name="test_app",
                api_endpoint="http://localhost:8080",
                telemetry_enabled=False,  # Explicit False
                exporter=mock.Mock()
            )
            
            # At this point, no Telemetry instance should exist and PostHog should not be called
            mock_posthog.assert_not_called()
            
            # 2. Now somewhere else in the code, Telemetry() gets called directly
            # This is the problem - it creates a singleton that ignores the init parameter
            # and only checks the environment variable (which defaults to "true")
            telemetry = Telemetry()
            
            # HERE IS THE BUG: The telemetry instance should be disabled because
            # Traceloop.init was called with telemetry_enabled=False, but it's not!
            # The Telemetry class only checks os.getenv("TRACELOOP_TELEMETRY") which defaults to "true"
            
            # This assertion should fail if the bug exists
            if telemetry._telemetry_enabled:
                # Bug exists - telemetry is enabled despite init parameter
                pytest.fail("BUG REPRODUCED: Telemetry ignores telemetry_enabled=False parameter from Traceloop.init()")
            
            # If we get here, telemetry should be properly disabled
            assert not telemetry._telemetry_enabled
            
            # Verify no PostHog calls were made
            mock_posthog.assert_not_called()

    def test_fix_validation(self):
        """Test that validates the fix - telemetry respects init parameter"""
        # Clear env var and reset Telemetry class state
        if 'TRACELOOP_TELEMETRY' in os.environ:
            del os.environ['TRACELOOP_TELEMETRY']
        
        # Reset the Telemetry singleton and global state
        from traceloop.sdk.telemetry import Telemetry
        if hasattr(Telemetry, 'instance'):
            delattr(Telemetry, 'instance')
        Telemetry._global_telemetry_enabled = None
            
        # Mock PostHog to track calls
        with mock.patch('traceloop.sdk.telemetry.Posthog') as mock_posthog:
            mock_posthog_instance = mock.Mock()
            mock_posthog.return_value = mock_posthog_instance
            
            # Initialize Traceloop with telemetry explicitly disabled
            Traceloop.init(
                app_name="test_app",
                api_endpoint="http://localhost:8080",
                telemetry_enabled=False,  # Explicit False
                exporter=mock.Mock()
            )
            
            # At this point, no telemetry should be happening
            mock_posthog.assert_not_called()
            
            # Now simulate what happens when an instrumentation calls Telemetry().capture()
            # This is the exact pattern from tracing.py:498
            telemetry = Telemetry()
            
            # After the fix, this should be False because init was called with telemetry_enabled=False
            assert not telemetry._telemetry_enabled, "Telemetry should respect init parameter telemetry_enabled=False"
            
            # Try to capture - should be a no-op
            telemetry.capture("instrumentation:test:init")
            
            # PostHog should never have been called
            mock_posthog.assert_not_called()