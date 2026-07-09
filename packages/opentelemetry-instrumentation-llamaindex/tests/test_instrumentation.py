import sys


def test_instrumentation_llamaindex():
    # clean installation of the instrumentation
    del sys.modules["opentelemetry.instrumentation.llamaindex"]
    from opentelemetry.instrumentation.llamaindex import LlamaIndexInstrumentor
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument()

    # assert the right sub-instrumentation is used
    assert instrumentor.is_instrumented_by_opentelemetry is True
    assert instrumentor.legacy.is_instrumented_by_opentelemetry is True
    assert instrumentor.core.is_instrumented_by_opentelemetry is False


def test_instrumentation_llamaindex_core(monkeypatch):
    # clean installation of otel
    imports = sys.modules.copy()
    for k, v in imports.items():
        if k.startswith("opentelemetry"):
            del sys.modules[k]
    # removal of llamaindex pkg
    import opentelemetry.instrumentation.dependencies

    def mock(deps):
        for dep in deps:
            req = opentelemetry.instrumentation.dependencies.Requirement(dep)
            if req.name == "llama-index":
                return opentelemetry.instrumentation.dependencies.DependencyConflict(dep)
        return None

    opentelemetry.instrumentation.dependencies.get_dependency_conflicts = mock

    # clean installation of the instrumentation
    from opentelemetry.instrumentation.llamaindex import LlamaIndexInstrumentor
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument()

    # assert the right sub-instrumentation is used
    assert instrumentor.is_instrumented_by_opentelemetry is True
    assert instrumentor.legacy.is_instrumented_by_opentelemetry is False
    assert instrumentor.core.is_instrumented_by_opentelemetry is True
