import asyncio

from opentelemetry.instrumentation.together.config import Config
from opentelemetry.instrumentation.together.utils import dont_throw


def test_dont_throw_swallows_errors_from_async_functions(monkeypatch):
    errors = []
    monkeypatch.setattr(Config, "exception_logger", errors.append)

    @dont_throw
    async def fails():
        raise ValueError("telemetry failure")

    assert asyncio.run(fails()) is None
    assert [str(error) for error in errors] == ["telemetry failure"]


def test_dont_throw_returns_the_result_of_async_functions():
    @dont_throw
    async def succeeds():
        return "parsed"

    assert asyncio.run(succeeds()) == "parsed"
