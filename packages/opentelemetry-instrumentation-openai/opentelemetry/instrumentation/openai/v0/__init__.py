from typing import Collection

from opentelemetry._events import get_event_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
    achat_wrapper,
    chat_wrapper,
)
from opentelemetry.instrumentation.openai.shared.completion_wrappers import (
    acompletion_wrapper,
    completion_wrapper,
)
from opentelemetry.instrumentation.openai.shared.config import Config
from opentelemetry.instrumentation.openai.shared.embeddings_wrappers import (
    aembeddings_wrapper,
    embeddings_wrapper,
)
from opentelemetry.instrumentation.openai.utils import is_metrics_enabled
from opentelemetry.instrumentation.openai.version import __version__
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.metrics import get_meter
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics as GenAIMetrics
from opentelemetry.semconv_ai import Meters
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

_instruments = ("openai >= 0.27.0", "openai < 1.0.0")


class OpenAIV0Instrumentor(BaseInstrumentor):
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        if not Config.use_legacy_attributes:
            event_logger_provider = kwargs.get("event_logger_provider")
            Config.event_logger = get_event_logger(
                __name__, __version__, event_logger_provider=event_logger_provider
            )

        if is_metrics_enabled():
            tokens_histogram = meter.create_histogram(
                name=Meters.LLM_TOKEN_USAGE,
                unit="token",
                description="Measures number of input and output tokens used",
            )

            chat_choice_counter = meter.create_counter(
                name=Meters.LLM_GENERATION_CHOICES,
                unit="choice",
                description="Number of choices returned by chat completions call",
            )

            duration_histogram = meter.create_histogram(
                name=Meters.LLM_OPERATION_DURATION,
                unit="s",
                description="GenAI operation duration",
            )

            chat_exception_counter = meter.create_counter(
                name=Meters.LLM_COMPLETIONS_EXCEPTIONS,
                unit="time",
                description="Number of exceptions occurred during chat completions",
            )

            streaming_time_to_first_token = meter.create_histogram(
                name=GenAIMetrics.GEN_AI_SERVER_TIME_TO_FIRST_TOKEN,
                unit="s",
                description="Time to first token in streaming chat completions",
            )
            streaming_time_to_generate = meter.create_histogram(
                name=Meters.LLM_STREAMING_TIME_TO_GENERATE,
                unit="s",
                description="Time between first token and completion in streaming chat completions",
            )
        else:
            (
                tokens_histogram,
                chat_choice_counter,
                duration_histogram,
                chat_exception_counter,
                streaming_time_to_first_token,
                streaming_time_to_generate,
            ) = (None, None, None, None, None, None)

        if is_metrics_enabled():
            embeddings_vector_size_counter = meter.create_counter(
                name=Meters.LLM_EMBEDDINGS_VECTOR_SIZE,
                unit="element",
                description="he size of returned vector",
            )
            embeddings_exception_counter = meter.create_counter(
                name=Meters.LLM_EMBEDDINGS_EXCEPTIONS,
                unit="time",
                description="Number of exceptions occurred during embeddings operation",
            )
        else:
            (
                tokens_histogram,
                embeddings_vector_size_counter,
                embeddings_exception_counter,
            ) = (None, None, None)

        wrap_function_wrapper(
            "openai",
            "Completion.create",
            completion_wrapper(tracer),
        )

        wrap_function_wrapper(
            "openai",
            "Completion.acreate",
            acompletion_wrapper(tracer),
        )
        wrap_function_wrapper(
            "openai",
            "ChatCompletion.create",
            chat_wrapper(
                tracer,
                tokens_histogram,
                chat_choice_counter,
                duration_histogram,
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
                duration_histogram,
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
                duration_histogram,
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
                duration_histogram,
                embeddings_exception_counter,
            ),
        )

    def _uninstrument(self, **kwargs):
        unwrap("openai", "Completion.create")
        unwrap("openai", "Completion.acreate")
        unwrap("openai", "ChatCompletion.create")
        unwrap("openai", "ChatCompletion.acreate")
        unwrap("openai", "Embedding.create")
        unwrap("openai", "Embedding.acreate")
