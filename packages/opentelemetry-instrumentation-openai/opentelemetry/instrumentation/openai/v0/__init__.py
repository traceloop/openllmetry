from typing import Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
    chat_wrapper,
    achat_wrapper,
)
from opentelemetry.instrumentation.openai.shared.completion_wrappers import (
    completion_wrapper,
    acompletion_wrapper,
)
from opentelemetry.instrumentation.openai.shared.embeddings_wrappers import (
    embeddings_wrapper,
    aembeddings_wrapper,
)
from opentelemetry.instrumentation.openai.version import __version__

_instruments = ("openai >= 0.27.0", "openai < 1.0.0")


class OpenAIV0Instrumentor(BaseInstrumentor):
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        wrap_function_wrapper("openai", "Completion.create", completion_wrapper(tracer))
        wrap_function_wrapper(
            "openai", "Completion.acreate", acompletion_wrapper(tracer)
        )
        wrap_function_wrapper("openai", "ChatCompletion.create", chat_wrapper(tracer))
        wrap_function_wrapper("openai", "ChatCompletion.acreate", achat_wrapper(tracer))
        wrap_function_wrapper("openai", "Embedding.create", embeddings_wrapper(tracer))
        wrap_function_wrapper(
            "openai", "Embedding.acreate", aembeddings_wrapper(tracer)
        )

    def _uninstrument(self, **kwargs):
        pass
