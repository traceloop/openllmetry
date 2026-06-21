"""Unit tests for langchain instrumentation utilities."""

import dataclasses
import json
import threading

from opentelemetry.instrumentation.langchain.utils import (
    CallbackFilteredJSONEncoder,
)


def test_encoder_serializes_plain_dataclass():
    @dataclasses.dataclass
    class Plain:
        name: str
        count: int

    result = json.loads(json.dumps(Plain(name="x", count=1), cls=CallbackFilteredJSONEncoder))

    assert result == {"name": "x", "count": 1}


def test_encoder_serializes_dataclass_with_rlock():
    @dataclasses.dataclass
    class WithLock:
        name: str
        lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)

    encoded = json.dumps(WithLock(name="x"), cls=CallbackFilteredJSONEncoder)
    result = json.loads(encoded)

    assert result["name"] == "x"
    assert isinstance(result["lock"], str)


def test_encoder_serializes_nested_dataclass_with_rlock():
    @dataclasses.dataclass
    class Inner:
        lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)

    @dataclasses.dataclass
    class Outer:
        name: str
        inner: Inner = dataclasses.field(default_factory=Inner)

    encoded = json.dumps(Outer(name="x"), cls=CallbackFilteredJSONEncoder)
    result = json.loads(encoded)

    assert result["name"] == "x"
    assert "lock" in result["inner"]


def test_encoder_serializes_list_of_dataclasses_under_fallback():
    @dataclasses.dataclass
    class Item:
        value: int

    @dataclasses.dataclass
    class Container:
        items: list
        lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)

    encoded = json.dumps(
        Container(items=[Item(1), Item(2)]), cls=CallbackFilteredJSONEncoder
    )
    result = json.loads(encoded)

    assert result["items"] == [{"value": 1}, {"value": 2}]
