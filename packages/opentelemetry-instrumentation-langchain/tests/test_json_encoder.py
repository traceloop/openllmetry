import dataclasses
import json
import threading

from pydantic import BaseModel

from opentelemetry.instrumentation.langchain.utils import CallbackFilteredJSONEncoder


class PydanticV2Model(BaseModel):
    name: str
    age: int


def test_pydantic_v2_uses_model_dump():
    model = PydanticV2Model(name="Alice", age=30)
    assert json.loads(json.dumps(model, cls=CallbackFilteredJSONEncoder)) == {"name": "Alice", "age": 30}


def test_pydantic_v2_no_double_encoding():
    # Regression: model_dump_json() would emit a JSON-encoded string literal.
    model = PydanticV2Model(name="Alice", age=30)
    parsed = json.loads(json.dumps(model, cls=CallbackFilteredJSONEncoder))
    assert isinstance(parsed, dict), "result must decode to a dict, not a string"


def test_default_strips_callbacks_from_dict():
    # The `isinstance(o, dict)` branch is only reachable when default() is
    # invoked directly (json.dumps natively handles dicts and their subclasses
    # without calling default()).
    data = {"callbacks": ["cb1"], "key": "value"}
    result = CallbackFilteredJSONEncoder().default(data)
    assert "callbacks" not in result
    assert result["key"] == "value"


def test_non_basemodel_with_dict_attribute_is_not_called():
    # The BaseModel gate prevents this from invoking the non-callable .dict attribute,
    # which would otherwise raise TypeError.
    class HasDictDataAttribute:
        dict = {"not": "callable"}

    result = json.dumps(HasDictDataAttribute(), cls=CallbackFilteredJSONEncoder)
    assert isinstance(json.loads(result), str)


def test_dataclass_with_rlock_uses_fieldwise_fallback():
    @dataclasses.dataclass
    class WithLock:
        name: str
        lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)

    result = json.loads(json.dumps(WithLock(name="x"), cls=CallbackFilteredJSONEncoder))

    assert result["name"] == "x"
    assert isinstance(result["lock"], str)


def test_dataclass_type_is_stringified_not_treated_as_an_instance():
    @dataclasses.dataclass
    class Plain:
        name: str

    result = json.loads(json.dumps(Plain, cls=CallbackFilteredJSONEncoder))

    assert isinstance(result, str)
    assert "Plain" in result


def test_nested_dataclass_with_rlock_uses_fallback_recursively():
    @dataclasses.dataclass
    class Inner:
        lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)

    @dataclasses.dataclass
    class Outer:
        name: str
        inner: Inner = dataclasses.field(default_factory=Inner)

    result = json.loads(json.dumps(Outer(name="x"), cls=CallbackFilteredJSONEncoder))

    assert result["name"] == "x"
    assert isinstance(result["inner"]["lock"], str)


def test_fallback_serializes_dataclass_lists_and_its_lock():
    @dataclasses.dataclass
    class Item:
        value: int

    @dataclasses.dataclass
    class Container:
        items: list[Item]
        lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)

    result = json.loads(json.dumps(Container(items=[Item(1), Item(2)]), cls=CallbackFilteredJSONEncoder))

    assert result["items"] == [{"value": 1}, {"value": 2}]
    assert isinstance(result["lock"], str)


def test_fallback_handles_non_typeerror_deepcopy_failures():
    class DeepcopyFailure:
        def __deepcopy__(self, memo):
            raise RuntimeError("not copyable")

    @dataclasses.dataclass
    class WithUncopyableField:
        value: DeepcopyFailure

    result = json.loads(json.dumps(WithUncopyableField(value=DeepcopyFailure()), cls=CallbackFilteredJSONEncoder))

    assert isinstance(result["value"], str)


def test_fallback_skips_dataclass_fields_that_raise_on_access():
    @dataclasses.dataclass
    class WithBrokenField:
        name: str
        broken: object = None
        lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)

        def __getattribute__(self, name):
            if name == "broken":
                raise RuntimeError("field access failed")
            return super().__getattribute__(name)

    result = json.loads(json.dumps(WithBrokenField(name="x"), cls=CallbackFilteredJSONEncoder))

    assert result["name"] == "x"
    assert "broken" not in result
    assert isinstance(result["lock"], str)
