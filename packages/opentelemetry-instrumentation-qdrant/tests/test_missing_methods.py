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


def _patch_fake_clients(monkeypatch):
    monkeypatch.setattr(
        qdrant_module,
        "qdrant_client",
        SimpleNamespace(
            QdrantClient=_FakeQdrantClient,
            AsyncQdrantClient=_FakeAsyncQdrantClient,
        ),
    )


def test_instrumentor_skips_methods_missing_from_client_class(monkeypatch):
    wrapped_paths = []

    _patch_fake_clients(monkeypatch)
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


def test_uninstrumentor_skips_methods_missing_from_client_class(monkeypatch):
    unwrapped_methods = []

    _patch_fake_clients(monkeypatch)
    monkeypatch.setattr(
        qdrant_module,
        "unwrap",
        lambda module_path, method: unwrapped_methods.append((module_path, method)),
    )

    qdrant_module.QdrantInstrumentor()._uninstrument()

    assert ("qdrant_client.QdrantClient", "upload_points") in unwrapped_methods
    assert ("qdrant_client.AsyncQdrantClient", "upload_points") in unwrapped_methods
    assert ("qdrant_client.QdrantClient", "upload_records") not in unwrapped_methods
    assert ("qdrant_client.AsyncQdrantClient", "upload_records") not in unwrapped_methods
