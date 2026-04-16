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
    yield
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
        instance_method = getattr(m, hook_name)
        class_method = getattr(m.__class__, hook_name)
        # Instance attribute should shadow the class method (instrumented)
        assert instance_method is not class_method.__get__(m, type(m)), (
            f"{hook_name} on instance should be instrumented (shadowed)"
        )
