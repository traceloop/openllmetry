"""Tests for JSONEncoder Pydantic v2 compatibility (issue #3516)."""

import dataclasses
import json
import warnings
import pytest

from traceloop.sdk.utils.json_encoder import JSONEncoder


class FakePydanticV2Model:
    """Simulates a Pydantic v2 BaseModel with model_dump()."""

    def __init__(self, value: int, name: str):
        self.value = value
        self.name = name

    def model_dump(self):
        return {"value": self.value, "name": self.name}

    def model_dump_json(self):
        return json.dumps({"value": self.value, "name": self.name})

    def json(self):
        warnings.warn(
            "The `json` method is deprecated; use `model_dump_json` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return json.dumps({"value": self.value, "name": self.name})


class FakePydanticV1Model:
    """Simulates a Pydantic v1 BaseModel with dict() and json()."""

    def __init__(self, value: int, name: str):
        self.value = value
        self.name = name

    def dict(self):
        return {"value": self.value, "name": self.name}

    def json(self):
        return json.dumps({"value": self.value, "name": self.name})


@dataclasses.dataclass
class SampleDataclass:
    x: int
    y: str


class ObjectWithToJson:
    def to_json(self):
        return {"custom": "json"}


class PlainObject:
    pass


def test_pydantic_v2_uses_model_dump():
    """Pydantic v2 objects should use model_dump() and return a dict."""
    encoder = JSONEncoder()
    model = FakePydanticV2Model(value=42, name="test")
    result = encoder.default(model)
    assert result == {"value": 42, "name": "test"}
    assert isinstance(result, dict)


def test_pydantic_v2_no_deprecation_warning():
    """Using model_dump() should NOT trigger the deprecated .json() warning."""
    encoder = JSONEncoder()
    model = FakePydanticV2Model(value=1, name="no_warning")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        # This should NOT raise a DeprecationWarning
        result = encoder.default(model)
    assert result == {"value": 1, "name": "no_warning"}


def test_pydantic_v1_uses_dict():
    """Pydantic v1 objects (without model_dump) should fall back to dict()."""
    encoder = JSONEncoder()
    model = FakePydanticV1Model(value=99, name="legacy")
    result = encoder.default(model)
    assert result == {"value": 99, "name": "legacy"}
    assert isinstance(result, dict)


def test_full_json_dumps_pydantic_v2():
    """End-to-end: json.dumps with JSONEncoder should serialize Pydantic v2 models."""
    model = FakePydanticV2Model(value=7, name="e2e")
    output = json.dumps(model, cls=JSONEncoder)
    parsed = json.loads(output)
    assert parsed == {"value": 7, "name": "e2e"}


def test_full_json_dumps_pydantic_v1():
    """End-to-end: json.dumps with JSONEncoder should serialize Pydantic v1 models."""
    model = FakePydanticV1Model(value=3, name="v1_e2e")
    output = json.dumps(model, cls=JSONEncoder)
    parsed = json.loads(output)
    assert parsed == {"value": 3, "name": "v1_e2e"}


def test_dataclass_encoding():
    """Dataclasses should still be handled correctly."""
    encoder = JSONEncoder()
    dc = SampleDataclass(x=10, y="hello")
    result = encoder.default(dc)
    assert result == {"x": 10, "y": "hello"}


def test_to_json_encoding():
    """Objects with to_json() should still be handled correctly."""
    encoder = JSONEncoder()
    obj = ObjectWithToJson()
    result = encoder.default(obj)
    assert result == {"custom": "json"}


def test_plain_object_fallback():
    """Objects with no special methods should fall back to __class__.__name__."""
    encoder = JSONEncoder()
    obj = PlainObject()
    result = encoder.default(obj)
    assert result == "PlainObject"
