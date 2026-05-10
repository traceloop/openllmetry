import dataclasses
import json
import warnings

import pytest

from traceloop.sdk.utils.json_encoder import JSONEncoder


# --- minimal Pydantic stubs (no real Pydantic dependency needed) ---

class PydanticV2Model:
    """Stub that mimics a Pydantic v2 model."""
    def model_dump(self):
        return {"name": "Alice", "age": 30}


class PydanticV1Model:
    """Stub that mimics a Pydantic v1 model (no model_dump)."""
    def dict(self):
        return {"name": "Bob", "age": 25}


class LegacyJsonModel:
    """Stub with only .json() — legacy path."""
    def json(self):
        return '{"name": "Charlie"}'


class ToJsonModel:
    """Stub with .to_json() — highest-priority path."""
    def to_json(self):
        return {"name": "Dave"}


@dataclasses.dataclass
class MyDataclass:
    x: int
    y: str


class PlainObject:
    pass


# --- tests ---

def test_pydantic_v2_uses_model_dump():
    result = json.dumps(PydanticV2Model(), cls=JSONEncoder)
    assert result == '{"name": "Alice", "age": 30}'


def test_pydantic_v2_no_double_encoding():
    # If model_dump_json() were used instead of model_dump(), the output would be
    # a JSON-encoded string literal: '"{\\"name\\": \\"Alice\\"}"'
    result = json.dumps(PydanticV2Model(), cls=JSONEncoder)
    parsed = json.loads(result)
    assert isinstance(parsed, dict), "result must decode to a dict, not a string"


def test_pydantic_v1_uses_dict():
    result = json.dumps(PydanticV1Model(), cls=JSONEncoder)
    assert result == '{"name": "Bob", "age": 25}'


def test_to_json_takes_priority_over_model_dump():
    class Both:
        def to_json(self):
            return {"source": "to_json"}
        def model_dump(self):
            return {"source": "model_dump"}

    result = json.loads(json.dumps(Both(), cls=JSONEncoder))
    assert result["source"] == "to_json"


def test_dataclass_serialization():
    result = json.loads(json.dumps(MyDataclass(x=1, y="hello"), cls=JSONEncoder))
    assert result == {"x": 1, "y": "hello"}


def test_plain_object_falls_back_to_class_name():
    result = json.dumps(PlainObject(), cls=JSONEncoder)
    assert result == '"PlainObject"'


def test_no_pydantic_deprecation_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Should not raise any DeprecationWarning
        json.dumps(PydanticV2Model(), cls=JSONEncoder)


def test_dict_with_callbacks_stripped():
    # default() is only called for non-natively-serializable objects,
    # so we wrap the dict in a custom object to trigger default()
    class DictWrapper:
        def __init__(self, d):
            self._d = d
        # no model_dump / dict / json — falls through to __class__ normally,
        # but we simulate the "isinstance dict" path via a subclass
    data = {"callbacks": ["cb1"], "key": "value"}
    # Directly invoke default() since plain dicts bypass it
    encoder = JSONEncoder()
    result = encoder.default(data)
    assert "callbacks" not in result
    assert result["key"] == "value"
