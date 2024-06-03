from typing import Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import get_tracer
from opentelemetry.metrics import get_meter
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
from opentelemetry.instrumentation.openai.utils import is_metrics_enabled
from opentelemetry.instrumentation.openai.version import __version__

_instruments = ("openai >= 0.27.0", "openai < 1.0.0")


class OpenAIV0Instrumentor(BaseInstrumentor):
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        if is_metrics_enabled():
            tokens_histogram = meter.create_histogram(
                name="gen_ai.client.token.usage",
                unit="token",
                description="Measures number of input and output tokens used",
            )

            chat_choice_counter = meter.create_counter(
                name="gen_ai.client.generation.choices",
                unit="choice",
                description="Number of choices returned by chat completions call",
            )

            chat_duration_histogram = meter.create_histogram(
                name="gen_ai.client.operation.duration",
                unit="s",
                description="GenAI operation duration",
            )

            chat_exception_counter = meter.create_counter(
                name="llm.openai.chat_completions.exceptions",
                unit="time",
                description="Number of exceptions occurred during chat completions",
            )

            streaming_time_to_first_token = meter.create_histogram(
                name="llm.openai.chat_completions.streaming_time_to_first_token",
                unit="s",
                description="Time to first token in streaming chat completions",
            )
            streaming_time_to_generate = meter.create_histogram(
                name="llm.openai.chat_completions.streaming_time_to_generate",
                unit="s",
                description="Time between first token and completion in streaming chat completions",
            )
        else:
            (
                tokens_histogram,
                chat_choice_counter,
                chat_duration_histogram,
                chat_exception_counter,
                streaming_time_to_first_token,
                streaming_time_to_generate,
            ) = (None, None, None, None, None, None)

        if is_metrics_enabled():
            embeddings_vector_size_counter = meter.create_counter(
                name="llm.openai.embeddings.vector_size",
                unit="element",
                description="he size of returned vector",
            )

            embeddings_duration_histogram = meter.create_histogram(
                name="llm.openai.embeddings.duration",
                unit="s",
                description="Duration of embeddings operation",
            )

            embeddings_exception_counter = meter.create_counter(
                name="llm.openai.embeddings.exceptions",
                unit="time",
                description="Number of exceptions occurred during embeddings operation",
            )
        else:
            (
                tokens_histogram,
                embeddings_vector_size_counter,
                embeddings_duration_histogram,
                embeddings_exception_counter,
            ) = (None, None, None, None)

        wrap_function_wrapper("openai", "Completion.create", completion_wrapper(tracer))
        wrap_function_wrapper(
            "openai", "Completion.acreate", acompletion_wrapper(tracer)
        )
        wrap_function_wrapper(
            "openai",
            "ChatCompletion.create",
            chat_wrapper(
                tracer,
                tokens_histogram,
                chat_choice_counter,
                chat_duration_histogram,
                chat_exception_counter,
                streaming_time_to_first_token,
                streaming_time_to_generate,
            ),
        )
        wrap_function_wrapper(
            "openai",
            "ChatCompletion.acreate",
            achat_wrapper(
                tracer,
                tokens_histogram,
                chat_choice_counter,
                chat_duration_histogram,
                chat_exception_counter,
                streaming_time_to_first_token,
                streaming_time_to_generate,
            ),
        )
        wrap_function_wrapper(
            "openai",
            "Embedding.create",
            embeddings_wrapper(
                tracer,
                tokens_histogram,
                embeddings_vector_size_counter,
                embeddings_duration_histogram,
                embeddings_exception_counter,
            ),
        )
        wrap_function_wrapper(
            "openai",
            "Embedding.acreate",
            aembeddings_wrapper(
                tracer,
                tokens_histogram,
                embeddings_vector_size_counter,
                embeddings_duration_histogram,
                embeddings_exception_counter,
            ),
        )

    def _uninstrument(self, **kwargs):
        pass
