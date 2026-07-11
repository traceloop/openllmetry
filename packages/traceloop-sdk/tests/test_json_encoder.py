import dataclasses
import json
import warnings

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


def test_legacy_json_method_fallback():
    # Regression test for #3516: .json() was the original deprecated path.
    # Objects exposing only .json() should still serialize without crashing.
    result = json.loads(json.dumps(LegacyJsonModel(), cls=JSONEncoder))
    assert result == {"name": "Charlie"}


def test_non_callable_dict_attribute_does_not_crash():
    # Many non-Pydantic objects (e.g., enum.DynamicClassAttribute) expose
    # a non-callable `.dict` attribute. The encoder must skip it instead
    # of raising TypeError.
    class HasDictDataAttribute:
        dict = {"not": "callable"}

    # Should fall through to __class__ name rather than raising.
    result = json.dumps(HasDictDataAttribute(), cls=JSONEncoder)
    assert result == '"HasDictDataAttribute"'


def test_coroutine_dict_method_does_not_crash():
    # Async dict() (e.g., some ORM models) returns a coroutine.
    # The encoder must skip and not emit un-encodable garbage.
    class AsyncDictModel:
        async def dict(self):
            return {"async": "result"}

    result = json.dumps(AsyncDictModel(), cls=JSONEncoder)
    assert result == '"AsyncDictModel"'


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


def test_default_strips_callbacks_from_dict():
    # The `isinstance(o, dict)` branch is only reachable when default() is
    # invoked directly (json.dumps natively handles dicts and their subclasses
    # without calling default()). Callers that hand a dict to .default()
    # explicitly — e.g., upstream callback-filtering glue — get callbacks stripped.
    data = {"callbacks": ["cb1"], "key": "value"}
    result = JSONEncoder().default(data)
    assert "callbacks" not in result
    assert result["key"] == "value"
