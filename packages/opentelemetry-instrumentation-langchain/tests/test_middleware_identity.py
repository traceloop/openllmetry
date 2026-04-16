"""Test that middleware hook instrumentation preserves base class identity checks.

LangGraph's create_agent uses identity checks like:
    m.__class__.before_agent is not AgentMiddleware.before_agent
to decide whether a middleware overrides a hook. Class-level wrapping with wrapt
breaks this by replacing base class methods with FunctionWrapper descriptors.
"""

import pytest
from langchain.agents.middleware.types import AgentMiddleware
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


ALL_HOOKS = [
    "before_model", "after_model", "before_agent", "after_agent",
    "abefore_model", "aafter_model", "abefore_agent", "aafter_agent",
]


class MyMiddleware(AgentMiddleware):
    """Subclass that only overrides before_agent."""

    def before_agent(self, state, runtime):
        return {"custom": True}


@pytest.fixture()
def _instrument():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    instrumentor = LangchainInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    yield instrumentor, exporter
    instrumentor.uninstrument()


def test_base_class_identity_preserved_after_instrumentation(_instrument):
    """Base class methods must remain identical via `is` after instrumentation.

    This is the exact check LangGraph's factory.py uses to decide whether
    to add graph nodes for middleware hooks.
    """
    for hook_name in ALL_HOOKS:
        base_method = getattr(AgentMiddleware, hook_name)
        sub_method = getattr(MyMiddleware, hook_name)
        if hook_name == "before_agent":
            # MyMiddleware overrides this — should NOT be identical
            assert sub_method is not base_method, (
                f"MyMiddleware.{hook_name} should differ from base"
            )
        else:
            # Not overridden — must be identical
            assert sub_method is base_method, (
                f"MyMiddleware.{hook_name} should be identical to "
                f"AgentMiddleware.{hook_name} but is not — "
                f"class-level wrapping likely broke identity"
            )


def test_instance_hooks_are_instrumented(_instrument):
    """Instance-level hooks should be wrapped for tracing after construction."""
    m = MyMiddleware()
    for hook_name in ALL_HOOKS:
        assert hook_name in m.__dict__, (
            f"{hook_name} should be in instance __dict__ (instrumented)"
        )


def test_uninstrument_removes_instance_patches(_instrument):
    """After uninstrument(), pre-existing instances must stop emitting spans."""
    instrumentor, exporter = _instrument

    m = MyMiddleware()
    # Verify hooks are patched
    assert "before_model" in m.__dict__, "Hook should be in instance __dict__"

    # Call a hook — should produce a span
    m.before_model({}, None)
    spans_before = exporter.get_finished_spans()
    assert len(spans_before) == 1

    exporter.clear()

    # Uninstrument — should clean up instance patches
    instrumentor.uninstrument()

    # Instance __dict__ should no longer shadow the hooks
    assert "before_model" not in m.__dict__, (
        "Hook should be removed from instance __dict__ after uninstrument"
    )

    # Calling the hook now goes to the unpatched class method — no span
    m.before_model({}, None)
    spans_after = exporter.get_finished_spans()
    assert len(spans_after) == 0, (
        "No spans should be emitted after uninstrument()"
    )
