"""OpenTelemetry Langchain instrumentation"""

import logging
from typing import Collection
from opentelemetry.instrumentation.langchain.config import Config
from opentelemetry.instrumentation.langchain.utils import is_metrics_enabled
from wrapt import wrap_function_wrapper
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.semconv_ai import Meters
from opentelemetry.trace import get_tracer
from opentelemetry.metrics import get_meter

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap

from opentelemetry.instrumentation.langchain.version import __version__


from opentelemetry.instrumentation.langchain.callback_handler import (
    TraceloopCallbackHandler,
)

logger = logging.getLogger(__name__)

_instruments = ("langchain >= 0.0.346", "langchain-core > 0.1.0")


class LangchainInstrumentor(BaseInstrumentor):
    """An instrumentor for Langchain SDK."""

    def __init__(self, exception_logger=None):
        super().__init__()
        Config.exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

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
                name=Meters.LLM_STREAMING_TIME_TO_FIRST_TOKEN,
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
            module="langchain_core.callbacks",
            name="BaseCallbackManager.__init__",
            wrapper=_BaseCallbackManagerInitWrapper(TraceloopCallbackHandler(
                tracer,)),
        )

    def _uninstrument(self, **kwargs):
        unwrap("langchain_core.callbacks", "BaseCallbackManager.__init__")


class _BaseCallbackManagerInitWrapper:
    def __init__(self, callback_manager: "TraceloopCallbackHandler"):
        self._callback_manager = callback_manager

    def __call__(
        self,
        wrapped,
        instance,
        args,
        kwargs,
    ) -> None:
        wrapped(*args, **kwargs)
        for handler in instance.inheritable_handlers:
            if isinstance(handler, type(self._callback_manager)):
                break
        else:
            instance.add_handler(self._callback_manager, True)
