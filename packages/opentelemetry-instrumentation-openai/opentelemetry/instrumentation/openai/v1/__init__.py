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
from opentelemetry.instrumentation.openai.shared.image_gen_wrappers import (
    image_gen_metrics_wrapper,
)
from opentelemetry.instrumentation.openai.utils import is_metrics_enabled
from opentelemetry.instrumentation.openai.v1.assistant_wrappers import (
    assistants_create_wrapper,
    messages_list_wrapper,
    runs_create_and_stream_wrapper,
    runs_create_wrapper,
    runs_retrieve_wrapper,
)

from opentelemetry.instrumentation.openai.v1.responses_wrappers import (
    async_responses_cancel_wrapper,
    async_responses_get_or_create_wrapper,
    responses_cancel_wrapper,
    responses_get_or_create_wrapper,
)

from opentelemetry.instrumentation.openai.version import __version__
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.metrics import get_meter
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics as GenAIMetrics
from opentelemetry.semconv_ai import Meters
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper


_instruments = ("openai >= 1.0.0",)


class OpenAIV1Instrumentor(BaseInstrumentor):
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _try_wrap(self, module, function, wrapper):
        """
        Wrap a function if it exists, otherwise do nothing.
        This is useful for handling cases where the function is not available in
        the older versions of the library.

        Args:
            module (str): The module to wrap, e.g. "openai.resources.chat.completions"
            function (str): "Object.function" to wrap, e.g. "Completions.parse"
            wrapper (callable): The wrapper to apply to the function.
        """
        try:
            wrap_function_wrapper(module, function, wrapper)
        except (AttributeError, ModuleNotFoundError):
            pass

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        # meter and counters are inited here
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

        wrap_function_wrapper(
            "openai.resources.chat.completions",
            "Completions.create",
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
            "openai.resources.completions",
            "Completions.create",
            completion_wrapper(tracer),
        )

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
            "openai.resources.embeddings",
            "Embeddings.create",
            embeddings_wrapper(
                tracer,
                tokens_histogram,
                embeddings_vector_size_counter,
                duration_histogram,
                embeddings_exception_counter,
            ),
        )

        wrap_function_wrapper(
            "openai.resources.chat.completions",
            "AsyncCompletions.create",
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
            "openai.resources.completions",
            "AsyncCompletions.create",
            acompletion_wrapper(tracer),
        )
        wrap_function_wrapper(
            "openai.resources.embeddings",
            "AsyncEmbeddings.create",
            aembeddings_wrapper(
                tracer,
                tokens_histogram,
                embeddings_vector_size_counter,
                duration_histogram,
                embeddings_exception_counter,
            ),
        )
        # in newer versions, Completions.parse are out of beta
        self._try_wrap(
            "openai.resources.chat.completions",
            "Completions.parse",
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
        self._try_wrap(
            "openai.resources.chat.completions",
            "AsyncCompletions.parse",
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

        if is_metrics_enabled():
            image_gen_exception_counter = meter.create_counter(
                name=Meters.LLM_IMAGE_GENERATIONS_EXCEPTIONS,
                unit="time",
                description="Number of exceptions occurred during image generations operation",
            )
        else:
            image_gen_exception_counter = None

        wrap_function_wrapper(
            "openai.resources.images",
            "Images.generate",
            image_gen_metrics_wrapper(duration_histogram, image_gen_exception_counter),
        )

        # Beta APIs may not be available consistently in all versions
        self._try_wrap(
            "openai.resources.beta.assistants",
            "Assistants.create",
            assistants_create_wrapper(tracer),
        )
        self._try_wrap(
            "openai.resources.beta.chat.completions",
            "Completions.parse",
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
        self._try_wrap(
            "openai.resources.beta.chat.completions",
            "AsyncCompletions.parse",
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
        self._try_wrap(
            "openai.resources.beta.threads.runs",
            "Runs.create",
            runs_create_wrapper(tracer),
        )
        self._try_wrap(
            "openai.resources.beta.threads.runs",
            "Runs.retrieve",
            runs_retrieve_wrapper(tracer),
        )
        self._try_wrap(
            "openai.resources.beta.threads.runs",
            "Runs.create_and_stream",
            runs_create_and_stream_wrapper(tracer),
        )
        self._try_wrap(
            "openai.resources.beta.threads.messages",
            "Messages.list",
            messages_list_wrapper(tracer),
        )
        self._try_wrap(
            "openai.resources.responses",
            "Responses.create",
            responses_get_or_create_wrapper(tracer),
        )
        self._try_wrap(
            "openai.resources.responses",
            "Responses.retrieve",
            responses_get_or_create_wrapper(tracer),
        )
        self._try_wrap(
            "openai.resources.responses",
            "Responses.cancel",
            responses_cancel_wrapper(tracer),
        )
        self._try_wrap(
            "openai.resources.responses",
            "AsyncResponses.create",
            async_responses_get_or_create_wrapper(tracer),
        )
        self._try_wrap(
            "openai.resources.responses",
            "AsyncResponses.retrieve",
            async_responses_get_or_create_wrapper(tracer),
        )
        self._try_wrap(
            "openai.resources.responses",
            "AsyncResponses.cancel",
            async_responses_cancel_wrapper(tracer),
        )

    def _uninstrument(self, **kwargs):
        unwrap("openai.resources.chat.completions", "Completions.create")
        unwrap("openai.resources.completions", "Completions.create")
        unwrap("openai.resources.embeddings", "Embeddings.create")
        unwrap("openai.resources.chat.completions", "AsyncCompletions.create")
        unwrap("openai.resources.completions", "AsyncCompletions.create")
        unwrap("openai.resources.embeddings", "AsyncEmbeddings.create")
        unwrap("openai.resources.images", "Images.generate")

        # Beta APIs may not be available consistently in all versions
        try:
            unwrap("openai.resources.beta.assistants", "Assistants.create")
            unwrap("openai.resources.beta.chat.completions", "Completions.parse")
            unwrap("openai.resources.beta.chat.completions", "AsyncCompletions.parse")
            unwrap("openai.resources.beta.threads.runs", "Runs.create")
            unwrap("openai.resources.beta.threads.runs", "Runs.retrieve")
            unwrap("openai.resources.beta.threads.runs", "Runs.create_and_stream")
            unwrap("openai.resources.beta.threads.messages", "Messages.list")
            unwrap("openai.resources.responses", "Responses.create")
            unwrap("openai.resources.responses", "Responses.retrieve")
            unwrap("openai.resources.responses", "Responses.cancel")
            unwrap("openai.resources.responses", "AsyncResponses.create")
            unwrap("openai.resources.responses", "AsyncResponses.retrieve")
            unwrap("openai.resources.responses", "AsyncResponses.cancel")
        except ImportError:
            pass
