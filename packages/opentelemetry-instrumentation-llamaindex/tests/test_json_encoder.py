import json

import pytest

from opentelemetry.instrumentation.llamaindex.utils import JSONEncoder


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
    # Pydantic v1-style objects exposing only .json() must serialize without
    # crashing or double-encoding.
    assert json.loads(json.dumps(LegacyJsonModel(), cls=JSONEncoder)) == {"name": "Charlie"}


def test_non_callable_json_attribute_does_not_crash():
    class HasJsonDataAttribute:
        json = {"not": "callable"}

    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps(HasJsonDataAttribute(), cls=JSONEncoder)


def test_non_callable_dict_attribute_does_not_crash_with_typeerror():
    # Non-Pydantic objects with a `.dict` data attribute would previously raise
    # TypeError (dict is not callable). The guard now skips and falls through
    # to super().default(), which raises TypeError for unknown types — but
    # critically NOT from trying to call a non-callable.
    class HasDictDataAttribute:
        dict = {"not": "callable"}

    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps(HasDictDataAttribute(), cls=JSONEncoder)


def test_coroutine_dict_method_does_not_emit_garbage():
    # Async dict() returns a coroutine; encoder must skip rather than emit
    # un-encodable garbage. Falls through to super().default().
    class AsyncDictModel:
        async def dict(self):
            return {"async": "result"}

    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps(AsyncDictModel(), cls=JSONEncoder)
