"""Regression test: OpenAIInstrumentor.uninstrument() must actually remove wrappers.

Previously _uninstrument called unwrap("module", "Class.method"), a form that
opentelemetry.instrumentation.utils.unwrap silently ignores (it treats the
first arg as a dotted path and getattr(module, "Class.method") never
resolves). Tracing stayed active after uninstrument(), and re-instrumenting
stacked a second wrapper.
"""
import wrapt
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.instrumentation.openai import OpenAIInstrumentor


def _is_wrapped(fn):
    return isinstance(fn, wrapt.BaseObjectProxy) and hasattr(fn, "__wrapped__")


def _targets():
    import openai.resources.chat.completions as chat_completions
    import openai.resources.embeddings as embeddings
    import openai.resources.responses as responses

    return [
        (chat_completions.Completions, "create"),
        (chat_completions.AsyncCompletions, "create"),
        (embeddings.Embeddings, "create"),
        (responses.Responses, "create"),
    ]


def test_uninstrument_removes_wrappers():
    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument(tracer_provider=TracerProvider())
    try:
        for cls, method in _targets():
            assert _is_wrapped(getattr(cls, method)), f"{cls.__name__}.{method} not wrapped"

        instrumentor.uninstrument()

        for cls, method in _targets():
            assert not _is_wrapped(getattr(cls, method)), (
                f"{cls.__name__}.{method} still wrapped after uninstrument()"
            )
    finally:
        # never leave wrappers behind for other tests
        instrumentor.uninstrument()


def test_reinstrument_does_not_stack_wrappers():
    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument(tracer_provider=TracerProvider())
    try:
        instrumentor.uninstrument()
        instrumentor.instrument(tracer_provider=TracerProvider())

        import openai.resources.chat.completions as chat_completions

        fn = chat_completions.Completions.create
        assert _is_wrapped(fn)
        # a stacked wrapper would itself wrap a proxy
        assert not _is_wrapped(fn.__wrapped__), "wrappers stacked on re-instrument"
    finally:
        instrumentor.uninstrument()
