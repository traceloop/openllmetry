from types import SimpleNamespace

from opentelemetry.instrumentation import qdrant as qdrant_module


class _DynamicMethodMeta(type):
    def __getattr__(cls, name):
        if name == "upload_records":
            return lambda *args, **kwargs: None
        raise AttributeError(name)


class _FakeQdrantClient(metaclass=_DynamicMethodMeta):
    def upload_points(self):
        return None

    def upload_collection(self):
        return None


class _FakeAsyncQdrantClient(_FakeQdrantClient):
    pass


def test_static_attribute_check_ignores_dynamic_class_lookup():
    assert hasattr(_FakeQdrantClient, "upload_records")
    assert not qdrant_module._has_static_attribute(_FakeQdrantClient, "upload_records")


def test_instrumentor_skips_methods_missing_from_client_class(monkeypatch):
    wrapped_paths = []

    monkeypatch.setattr(
        qdrant_module,
        "qdrant_client",
        SimpleNamespace(
            QdrantClient=_FakeQdrantClient,
            AsyncQdrantClient=_FakeAsyncQdrantClient,
        ),
    )
    monkeypatch.setattr(qdrant_module, "get_tracer", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        qdrant_module,
        "wrap_function_wrapper",
        lambda module, path, wrapper: wrapped_paths.append(path),
    )

    qdrant_module.QdrantInstrumentor()._instrument()

    assert "QdrantClient.upload_points" in wrapped_paths
    assert "AsyncQdrantClient.upload_points" in wrapped_paths
    assert "QdrantClient.upload_records" not in wrapped_paths
    assert "AsyncQdrantClient.upload_records" not in wrapped_paths
