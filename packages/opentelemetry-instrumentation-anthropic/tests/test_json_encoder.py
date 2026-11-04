import json

import pytest

from opentelemetry.instrumentation.anthropic.utils import JSONEncoder


class PydanticV2Model:
    def model_dump(self):
        return {"name": "Alice", "age": 30}


class PydanticV1Model:
    def dict(self):
        return {"name": "Bob", "age": 25}


def test_pydantic_v2_uses_model_dump():
    assert json.loads(json.dumps(PydanticV2Model(), cls=JSONEncoder)) == {"name": "Alice", "age": 30}


def test_pydantic_v2_no_double_encoding():
    # Regression: model_dump_json() would emit a JSON-encoded string literal.
    parsed = json.loads(json.dumps(PydanticV2Model(), cls=JSONEncoder))
    assert isinstance(parsed, dict), "result must decode to a dict, not a string"


def test_pydantic_v1_uses_dict():
    assert json.loads(json.dumps(PydanticV1Model(), cls=JSONEncoder)) == {"name": "Bob", "age": 25}


@pytest.mark.parametrize("model_factory", [
    pytest.param(lambda: type("HasDictDataAttribute", (), {"dict": {"not": "callable"}})(),
                 id="non-callable-dict"),
])
def test_non_pydantic_dict_attribute_does_not_crash(model_factory):
    # Non-Pydantic objects with a `.dict` attribute must not raise TypeError.
    # The anthropic encoder falls back to str(o) — just assert it doesn't crash.
    result = json.dumps(model_factory(), cls=JSONEncoder)
    assert isinstance(json.loads(result), str)


def test_coroutine_dict_method_does_not_crash():
    class AsyncDictModel:
        async def dict(self):
            return {"async": "result"}

    # Must not emit un-encodable garbage; falls back to str(o).
    result = json.dumps(AsyncDictModel(), cls=JSONEncoder)
    assert isinstance(json.loads(result), str)
