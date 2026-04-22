import opentelemetry.instrumentation.langchain as langchain_module
from opentelemetry.instrumentation.langchain import LangchainInstrumentor


def test_instrumentation_uses_wrapt_positional_signature(monkeypatch):
    calls = []

    def strict_wrap_function(module, name, wrapper, /):
        calls.append((module, name, wrapper))

    monkeypatch.setattr(
        langchain_module, "wrap_function_wrapper", strict_wrap_function
    )
    monkeypatch.setattr(
        langchain_module, "is_package_available", lambda package_name: False
    )

    LangchainInstrumentor()._instrument()

    assert len(calls) == 1
    assert calls[0][0] == "langchain_core.callbacks"
    assert calls[0][1] == "BaseCallbackManager.__init__"


def test_uninstrumentation_unwraps_base_chat_openai(monkeypatch):
    calls = []

    def capture_unwrap(module, name):
        calls.append((module, name))

    monkeypatch.setattr(langchain_module, "unwrap", capture_unwrap)
    monkeypatch.setattr(
        langchain_module,
        "is_package_available",
        lambda package_name: package_name == "langchain_openai",
    )

    LangchainInstrumentor()._uninstrument()

    assert (
        "langchain_openai.chat_models.base",
        "BaseChatOpenAI._generate",
    ) in calls
    assert (
        "langchain_openai.chat_models.base",
        "BaseChatOpenAI._agenerate",
    ) in calls
