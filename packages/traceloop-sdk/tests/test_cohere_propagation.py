import pytest
from unittest import mock

from traceloop.sdk import Traceloop
from traceloop.sdk.tracing import tracing
from traceloop.sdk.tracing.tracing import TracerWrapper  # Import TracerWrapper


def test_propagate_use_legacy_attributes_cohere_true():
    with mock.patch("traceloop.sdk.tracing.tracing.init_cohere_instrumentor") as mock_init_cohere:
        # Explicitly reset the TracerWrapper singleton
        if hasattr(TracerWrapper, "instance"):
            del TracerWrapper.instance
        Traceloop.init(app_name="test_app", use_legacy_attributes=True, exporter=mock.Mock())  # ADD exporter=mock.Mock()
        mock_init_cohere.assert_called_once_with(use_legacy_attributes=True)
        Traceloop.instance = None  # Reset singleton

def test_propagate_use_legacy_attributes_cohere_false():
    with mock.patch("traceloop.sdk.tracing.tracing.init_cohere_instrumentor") as mock_init_cohere:
        # Explicitly reset the TracerWrapper singleton
        if hasattr(TracerWrapper, "instance"):
            del TracerWrapper.instance
        Traceloop.init(app_name="test_app", use_legacy_attributes=False, exporter=mock.Mock()) # ADD exporter=mock.Mock()
        mock_init_cohere.assert_called_once_with(use_legacy_attributes=False)
        Traceloop.instance = None  # Reset singleton

def test_propagate_use_legacy_attributes_cohere_default():
    # Assuming the default value in your Config is True
    with mock.patch("traceloop.sdk.tracing.tracing.init_cohere_instrumentor") as mock_init_cohere:
        # Explicitly reset the TracerWrapper singleton
        if hasattr(TracerWrapper, "instance"):
            del TracerWrapper.instance
        Traceloop.init(app_name="test_app", exporter=mock.Mock()) # ADD exporter=mock.Mock()
        mock_init_cohere.assert_called_once_with(use_legacy_attributes=True)
        Traceloop.instance = None  # Reset singleton