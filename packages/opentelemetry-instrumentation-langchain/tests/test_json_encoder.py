import json

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
