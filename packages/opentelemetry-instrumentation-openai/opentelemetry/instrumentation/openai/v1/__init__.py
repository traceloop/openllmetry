from typing import Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.openai.shared.chat_wrappers import chat_wrapper, achat_wrapper
from opentelemetry.instrumentation.openai.shared.completion_wrappers import completion_wrapper, acompletion_wrapper
from opentelemetry.instrumentation.openai.version import __version__

_instruments = ("openai >= 1.0.0",)


class OpenAIV1Instrumentor(BaseInstrumentor):
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        wrap_function_wrapper("openai.resources.chat.completions", "Completions.create", chat_wrapper(tracer))
        wrap_function_wrapper("openai.resources.completions", "Completions.create", completion_wrapper(tracer))
        wrap_function_wrapper("openai.resources.chat.completions", "AsyncCompletions.create", achat_wrapper(tracer))
        wrap_function_wrapper("openai.resources.completions", "AsyncCompletions.create", acompletion_wrapper(tracer))

    def _uninstrument(self, **kwargs):
        pass
