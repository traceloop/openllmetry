import os
from unittest import mock
import pytest
from traceloop.sdk import Traceloop
from traceloop.sdk.telemetry import Telemetry


class TestTelemetry:
    def setup_method(self):
        if hasattr(Telemetry, 'instance'):
            delattr(Telemetry, 'instance')

        Telemetry._global_telemetry_enabled = None

        if 'TRACELOOP_TELEMETRY' in os.environ:
            del os.environ['TRACELOOP_TELEMETRY']

    def teardown_method(self):
        if hasattr(Telemetry, 'instance'):
            delattr(Telemetry, 'instance')

        Telemetry._global_telemetry_enabled = None

        if 'TRACELOOP_TELEMETRY' in os.environ:
            del os.environ['TRACELOOP_TELEMETRY']

    def test_telemetry_disabled_via_init_parameter(self):
        """Test that telemetry_enabled=False in Traceloop.init() should disable telemetry"""
        with mock.patch('traceloop.sdk.telemetry.Posthog') as mock_posthog:
            Traceloop.init(
                app_name="test_app",
                api_endpoint="http://localhost:8080",
                telemetry_enabled=False,
                exporter=mock.Mock()
            )

            mock_posthog.assert_not_called()

            telemetry = Telemetry()
            assert not telemetry._telemetry_enabled, "Telemetry should be disabled when telemetry_enabled=False"

    def test_telemetry_disabled_in_pytest(self):
        """Test that telemetry is disabled when running in pytest"""
        with mock.patch('traceloop.sdk.telemetry.Posthog') as mock_posthog:
            Traceloop.init(
                app_name="test_app",
                api_endpoint="http://localhost:8080",
                exporter=mock.Mock()
            )

            mock_posthog.assert_not_called()

            telemetry = Telemetry()
            assert not telemetry._telemetry_enabled, "Telemetry should be disabled when running in pytest"

    def test_telemetry_disabled_via_env_var(self):
        """Test that TRACELOOP_TELEMETRY=false disables telemetry"""
        os.environ['TRACELOOP_TELEMETRY'] = 'false'

        with mock.patch('traceloop.sdk.telemetry.Posthog') as mock_posthog:
            Traceloop.init(
                app_name="test_app",
                api_endpoint="http://localhost:8080",
                exporter=mock.Mock()
            )

            mock_posthog.assert_not_called()

            telemetry = Telemetry()
            assert not telemetry._telemetry_enabled, "Telemetry should be disabled when TRACELOOP_TELEMETRY=false"

    def test_init_parameter_overrides_env_var(self):
        """Test that telemetry_enabled parameter should override environment variable"""
        os.environ['TRACELOOP_TELEMETRY'] = 'true'

        with mock.patch('traceloop.sdk.telemetry.Posthog') as mock_posthog:
            Traceloop.init(
                app_name="test_app",
                api_endpoint="http://localhost:8080",
                telemetry_enabled=False,
                exporter=mock.Mock()
            )

            mock_posthog.assert_not_called()

    def test_telemetry_bug_reproduction_real_scenario(self):
        """Test the actual scenario described in GitHub issue #3175"""
        if 'TRACELOOP_TELEMETRY' in os.environ:
            del os.environ['TRACELOOP_TELEMETRY']

        with mock.patch('traceloop.sdk.telemetry.Posthog') as mock_posthog:
            mock_posthog_instance = mock.Mock()
            mock_posthog.return_value = mock_posthog_instance

            Traceloop.init(
                app_name="test_app",
                api_endpoint="http://localhost:8080",
                telemetry_enabled=False,
                exporter=mock.Mock()
            )

            mock_posthog.assert_not_called()

            telemetry = Telemetry()

            if telemetry._telemetry_enabled:
                pytest.fail("BUG REPRODUCED: Telemetry ignores telemetry_enabled=False parameter from Traceloop.init()")

            assert not telemetry._telemetry_enabled

            mock_posthog.assert_not_called()

    def test_fix_validation(self):
        """Test that validates the fix - telemetry respects init parameter"""
        if 'TRACELOOP_TELEMETRY' in os.environ:
            del os.environ['TRACELOOP_TELEMETRY']

        from traceloop.sdk.telemetry import Telemetry
        if hasattr(Telemetry, 'instance'):
            delattr(Telemetry, 'instance')
        Telemetry._global_telemetry_enabled = None

        with mock.patch('traceloop.sdk.telemetry.Posthog') as mock_posthog:
            mock_posthog_instance = mock.Mock()
            mock_posthog.return_value = mock_posthog_instance

            Traceloop.init(
                app_name="test_app",
                api_endpoint="http://localhost:8080",
                telemetry_enabled=False,
                exporter=mock.Mock()
            )

            mock_posthog.assert_not_called()

            telemetry = Telemetry()

            assert not telemetry._telemetry_enabled, "Telemetry should respect init parameter telemetry_enabled=False"

            telemetry.capture("instrumentation:test:init")

            mock_posthog.assert_not_called()
