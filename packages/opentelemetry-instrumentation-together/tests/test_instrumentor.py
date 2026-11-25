import importlib

from opentelemetry.instrumentation.together import (
    WRAPPED_METHODS,
    TogetherAiInstrumentor,
)
from wrapt import ObjectProxy


def _installed_create_methods():
    """`create` as defined on every WRAPPED_METHODS class in the installed together version."""
    methods = {}
    for wrapped_method in WRAPPED_METHODS:
        module = importlib.import_module(wrapped_method["module"])
        cls = getattr(module, wrapped_method["object"], None)
        if cls is not None:
            methods[f"{wrapped_method['module']}.{wrapped_method['object']}"] = cls.__dict__["create"]
    return methods


def test_uninstrument_unwraps_every_wrapped_method(tracer_provider):
    instrumentor = TogetherAiInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        methods = _installed_create_methods()
        # Sync and async chat and completion in every together version.
        assert len(methods) == 4
        assert all(isinstance(method, ObjectProxy) for method in methods.values())
    finally:
        instrumentor.uninstrument()

    wrapped_after = [name for name, method in _installed_create_methods().items() if isinstance(method, ObjectProxy)]
    assert wrapped_after == []
