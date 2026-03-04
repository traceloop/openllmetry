"""Tests that instrumentation survives Cohere SDK's __init__ method overrides.

Cohere SDK's Client and AsyncClient __init__ methods overwrite self.chat and
self.chat_stream with experimental_kwarg_decorator wrappers.  These tests
verify that our __init__ wrapper correctly removes those instance-level
overrides so that the class-level wrapt wrappers remain effective.
"""

import os

import cohere
import pytest
from opentelemetry.instrumentation.cohere import CohereInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture(autouse=True)
def environment():
    if "COHERE_API_KEY" not in os.environ:
        os.environ["COHERE_API_KEY"] = "test_api_key"


@pytest.fixture
def tracer_provider():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


@pytest.fixture
def instrumentor(tracer_provider):
    inst = CohereInstrumentor()
    inst.instrument(tracer_provider=tracer_provider)
    yield inst
    inst.uninstrument()


def test_client_chat_not_shadowed_by_instance_attribute(instrumentor):
    """Client.chat should go through wrapt wrapper, not instance attribute."""
    client = cohere.Client("test_key")
    assert "chat" not in client.__dict__, (
        "chat should not be an instance attribute after instrumentation"
    )
    assert "BoundFunctionWrapper" in type(client.chat).__name__


def test_client_chat_stream_not_shadowed_by_instance_attribute(instrumentor):
    """Client.chat_stream should go through wrapt wrapper, not instance attribute."""
    client = cohere.Client("test_key")
    assert "chat_stream" not in client.__dict__, (
        "chat_stream should not be an instance attribute after instrumentation"
    )
    assert "BoundFunctionWrapper" in type(client.chat_stream).__name__


def test_async_client_chat_not_shadowed_by_instance_attribute(instrumentor):
    """AsyncClient.chat should go through wrapt wrapper, not instance attribute."""
    client = cohere.AsyncClient("test_key")
    assert "chat" not in client.__dict__, (
        "chat should not be an instance attribute after instrumentation"
    )
    assert "BoundFunctionWrapper" in type(client.chat).__name__


def test_async_client_chat_stream_not_shadowed_by_instance_attribute(instrumentor):
    """AsyncClient.chat_stream should go through wrapt wrapper, not instance attribute."""
    client = cohere.AsyncClient("test_key")
    assert "chat_stream" not in client.__dict__, (
        "chat_stream should not be an instance attribute after instrumentation"
    )
    assert "BoundFunctionWrapper" in type(client.chat_stream).__name__


def test_client_v2_chat_not_shadowed_by_instance_attribute(instrumentor):
    """ClientV2.chat should go through wrapt wrapper, not instance attribute."""
    client = cohere.ClientV2("test_key")
    assert "chat" not in client.__dict__, (
        "chat should not be an instance attribute after instrumentation"
    )
    assert "BoundFunctionWrapper" in type(client.chat).__name__


def test_async_client_v2_chat_not_shadowed_by_instance_attribute(instrumentor):
    """AsyncClientV2.chat should go through wrapt wrapper, not instance attribute."""
    client = cohere.AsyncClientV2("test_key")
    assert "chat" not in client.__dict__, (
        "chat should not be an instance attribute after instrumentation"
    )
    assert "BoundFunctionWrapper" in type(client.chat).__name__


def test_non_overwritten_methods_still_wrapped(instrumentor):
    """Methods not affected by experimental_kwarg_decorator should still be wrapped."""
    client = cohere.Client("test_key")
    assert "BoundFunctionWrapper" in type(client.rerank).__name__
    assert "BoundFunctionWrapper" in type(client.embed).__name__
    assert "BoundFunctionWrapper" in type(client.generate).__name__


def test_uninstrument_restores_original(tracer_provider):
    """After uninstrument, __init__ should no longer be wrapped."""
    inst = CohereInstrumentor()
    inst.instrument(tracer_provider=tracer_provider)
    inst.uninstrument()

    # After uninstrument, creating a new client should have the instance
    # attributes back (from experimental_kwarg_decorator)
    client = cohere.Client("test_key")
    # The methods should be regular functions, not FunctionWrappers
    assert "FunctionWrapper" not in type(client.rerank).__name__
