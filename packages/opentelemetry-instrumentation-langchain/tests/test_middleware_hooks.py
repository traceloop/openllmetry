"""Instrumenting must not change how ``create_agent`` treats middleware hooks."""

import asyncio
import inspect

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage

HOOKS = [
    "before_model",
    "after_model",
    "before_agent",
    "after_agent",
    "abefore_model",
    "aafter_model",
    "abefore_agent",
    "aafter_agent",
]

REQUEST = {"messages": [{"role": "user", "content": "hi"}]}


def _agent(*middleware):
    model = FakeMessagesListChatModel(responses=[AIMessage(content="done")])
    return create_agent(model, tools=[], middleware=list(middleware))


class SyncOnly(AgentMiddleware):
    runtimes = []

    def before_model(self, state, runtime):
        SyncOnly.runtimes.append(runtime)


class AsyncOnly(AgentMiddleware):
    async def abefore_model(self, state, runtime):
        return None


class Both(AgentMiddleware):
    runtimes = []

    def before_model(self, state, runtime):
        Both.runtimes.append(("sync", runtime))

    async def abefore_model(self, state, runtime):
        Both.runtimes.append(("async", runtime))


@pytest.mark.parametrize("hook", HOOKS)
def test_base_hook_identity_is_stable(instrument_legacy, hook):
    # create_agent decides whether a middleware overrides a hook with
    # `cls.hook is not AgentMiddleware.hook`, so the base attribute must
    # resolve to the same object on every access.
    assert getattr(AgentMiddleware, hook) is getattr(AgentMiddleware, hook)


@pytest.mark.parametrize("hook", HOOKS)
def test_base_hook_keeps_signature(instrument_legacy, hook):
    parameters = list(inspect.signature(getattr(AgentMiddleware, hook)).parameters)
    assert parameters == ["self", "state", "runtime"]


@pytest.mark.parametrize("hook", HOOKS)
def test_base_hook_keeps_sync_or_async_kind(instrument_legacy, hook):
    is_async = inspect.iscoroutinefunction(getattr(AgentMiddleware, hook))
    assert is_async == hook.startswith(("abefore", "aafter"))


def test_middleware_without_hooks_adds_no_graph_nodes(instrument_legacy):
    from langchain.agents.middleware import wrap_model_call

    @wrap_model_call
    def passthrough(request, handler):
        return handler(request)

    nodes = sorted(_agent(passthrough).get_graph().nodes)
    assert nodes == ["__end__", "__start__", "model"]


def test_sync_only_hook_runs_under_invoke_and_receives_runtime(instrument_legacy):
    SyncOnly.runtimes.clear()
    _agent(SyncOnly()).invoke(REQUEST)
    assert len(SyncOnly.runtimes) == 1
    assert SyncOnly.runtimes[0] is not None


def test_sync_only_hook_still_runs_under_ainvoke(instrument_legacy):
    SyncOnly.runtimes.clear()
    asyncio.run(_agent(SyncOnly()).ainvoke(REQUEST))
    assert len(SyncOnly.runtimes) == 1


def test_async_only_hook_is_not_silently_skipped_under_invoke(instrument_legacy):
    # Without tracing LangChain refuses to run an async-only hook from the
    # sync entry point. Tracing must not turn that error into a silent skip.
    with pytest.raises(Exception):
        _agent(AsyncOnly()).invoke(REQUEST)


@pytest.mark.parametrize("use_async", [False, True])
def test_hooks_receive_runtime_when_both_variants_exist(instrument_legacy, use_async):
    Both.runtimes.clear()
    agent = _agent(Both())
    if use_async:
        asyncio.run(agent.ainvoke(REQUEST))
    else:
        agent.invoke(REQUEST)
    assert [kind for kind, _ in Both.runtimes] == ["async" if use_async else "sync"]
    assert Both.runtimes[0][1] is not None


class NoOverrides(AgentMiddleware):
    pass


def test_base_hooks_are_still_traced_when_called_directly(
    instrument_legacy, span_exporter
):
    span_exporter.clear()
    middleware = NoOverrides()

    middleware.before_model({}, None)
    asyncio.run(middleware.abefore_model({}, None))

    names = [span.name for span in span_exporter.get_finished_spans()]
    assert "execute_task NoOverrides.before_model" in names
    assert "execute_task NoOverrides.abefore_model" in names
