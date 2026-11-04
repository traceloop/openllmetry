import json

from opentelemetry.instrumentation.openai_agents.utils import JSONEncoder


class PydanticV2Model:
    def model_dump(self):
        return {"name": "Alice", "age": 30}


class PydanticV1Model:
    def dict(self):
        return {"name": "Bob", "age": 25}


class LegacyJsonModel:
    def json(self):
        return '{"name": "Charlie"}'


def test_pydantic_v2_uses_model_dump():
    assert json.loads(json.dumps(PydanticV2Model(), cls=JSONEncoder)) == {"name": "Alice", "age": 30}


def test_pydantic_v2_no_double_encoding():
    # Regression: model_dump_json() would emit a JSON-encoded string literal.
    parsed = json.loads(json.dumps(PydanticV2Model(), cls=JSONEncoder))
    assert isinstance(parsed, dict), "result must decode to a dict, not a string"


def test_pydantic_v1_uses_dict():
    assert json.loads(json.dumps(PydanticV1Model(), cls=JSONEncoder)) == {"name": "Bob", "age": 25}


def test_legacy_json_method_fallback():
    assert json.loads(json.dumps(LegacyJsonModel(), cls=JSONEncoder)) == {"name": "Charlie"}


def test_non_callable_dict_attribute_does_not_crash():
    class HasDictDataAttribute:
        dict = {"not": "callable"}

    result = json.dumps(HasDictDataAttribute(), cls=JSONEncoder)
    assert result == '"HasDictDataAttribute"'


def test_coroutine_dict_method_does_not_crash():
    class AsyncDictModel:
        async def dict(self):
            return {"async": "result"}

    result = json.dumps(AsyncDictModel(), cls=JSONEncoder)
    assert result == '"AsyncDictModel"'


def test_non_callable_json_attribute_does_not_crash():
    class HasJsonDataAttribute:
        json = {"not": "callable"}

    result = json.dumps(HasJsonDataAttribute(), cls=JSONEncoder)
    assert result == '"HasJsonDataAttribute"'
