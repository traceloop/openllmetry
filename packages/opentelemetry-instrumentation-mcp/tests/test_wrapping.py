"""Teardown must preserve callable behavior and other users' wrappers."""

from functools import wraps
import sys
from types import ModuleType

import pytest
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.mcp._wrapping import _InstrumentedMethods


@pytest.fixture
def module(monkeypatch):
    target = ModuleType("mcp_wrapping_test")
    monkeypatch.setitem(sys.modules, target.__name__, target)
    return target


def _record(events, label):
    def wrapper(wrapped, instance, args, kwargs):
        events.append(label)
        return wrapped(*args, **kwargs)
    return wrapper


@pytest.mark.parametrize("other_position", ["before", "after", "decorator", "opaque"])
def test_preserve_other_wrappers(module, other_position):
    events = []
    failure = RuntimeError("original failure")

    def original(fail=False):
        events.append("original")
        if fail:
            raise failure
        return 8

    module.call = original
    if other_position == "before":
        other = wrap_function_wrapper(module, "call", _record(events, "other"))
    methods = _InstrumentedMethods()
    methods.wrap(module.__name__, "call", _record(events, "ours"))
    cached = module.call
    if other_position == "after":
        other = wrap_function_wrapper(module, "call", _record(events, "other"))
    elif other_position in ("decorator", "opaque"):
        def other(*args, **kwargs):
            events.append("other")
            return cached(*args, **kwargs)
        if other_position == "decorator":
            other = wraps(cached)(other)
        module.call = other

    assert module.call() == 8
    assert sorted(events) == ["original", "other", "ours"]
    methods.unwrap()
    assert module.call is other
    events.clear()
    assert module.call() == 8
    assert events == ["other", "original"]
    events.clear()
    assert cached() == 8
    assert "ours" not in events
    with pytest.raises(RuntimeError) as caught:
        module.call(fail=True)
    assert caught.value is failure

    methods.wrap(module.__name__, "call", _record(events, "new"))
    events.clear()
    assert module.call() == 8
    assert sorted(events) == ["new", "original", "other"]
    methods.unwrap()
    assert module.call is other
    if other_position in ("before", "after"):
        assert module.call.__wrapped__ is original


async def test_cached_bound_method_uses_current_wrapper(module):
    class Client:
        async def call(self, value):
            assert self is client
            return value

    module.Client = Client
    client = Client()
    original = vars(Client)["call"]
    methods = _InstrumentedMethods()
    events = []
    cached = None
    for label in ("first", "second"):
        methods.wrap(module.__name__, "Client.call", _record(events, label))
        if cached is None:
            cached = client.call
        events.clear()
        assert await cached(8) == 8
        assert events == [label]
        methods.unwrap()
        assert vars(Client)["call"] is original
        events.clear()
        assert await cached(8) == 8
        assert not events


@pytest.mark.parametrize("expose_wrapped", [False, True])
def test_external_class_decorator_does_not_duplicate_tracing(module, expose_wrapped):
    class Client:
        def call(self, value):
            return value

    module.Client = Client
    methods = _InstrumentedMethods()
    events = []
    traced = _record(events, "traced")
    methods.wrap(module.__name__, "Client.call", traced)
    cached = Client.call  # A BoundFunctionWrapper, distinct from vars(Client)["call"].

    def other(self, value):
        return cached(self, value)

    Client.call = wraps(cached)(other) if expose_wrapped else other
    methods.unwrap()
    methods.wrap(module.__name__, "Client.call", traced)
    assert Client().call(8) == 8
    assert events == ["traced"]
    methods.unwrap()
    assert Client.call is other


def test_replaced_target_does_not_redirect_cached_reference(module):
    module.call = lambda: 8
    methods = _InstrumentedMethods()
    events = []
    traced = _record(events, "traced")
    methods.wrap(module.__name__, "call", traced)
    cached = module.call
    methods.unwrap()
    module.call = lambda: 9
    replacement = module.call
    methods.wrap(module.__name__, "call", traced)
    assert cached() == 8
    events.clear()
    assert module.call() == 9
    assert events == ["traced"]
    methods.unwrap()
    assert module.call is replacement


def test_deleted_target_does_not_abort_other_cleanup(module):
    module.first = lambda: 8
    module.second = lambda: 9
    original = module.first
    methods = _InstrumentedMethods()
    for name in ("first", "second"):
        methods.wrap(module.__name__, name, _record([], "traced"))
    del module.second
    methods.unwrap()
    assert not hasattr(module, "second")
    assert module.first is original
