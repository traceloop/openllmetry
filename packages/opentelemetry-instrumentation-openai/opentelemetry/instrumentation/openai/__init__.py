from typing import Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

from opentelemetry.instrumentation.openai.shared.config import Config
from opentelemetry.instrumentation.openai.utils import is_openai_v1
from opentelemetry.instrumentation.openai.v0 import OpenAIV0Instrumentor
from opentelemetry.instrumentation.openai.v1 import OpenAIV1Instrumentor

_instruments = ("openai >= 0.27.0",)


class OpenAIInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI's client library."""

    def __init__(
        self,
        enrich_assistant: bool = False,
        enrich_token_usage: bool = False,
        exception_logger=None,
    ):
        super().__init__()
        Config.enrich_assistant = enrich_assistant
        Config.enrich_token_usage = enrich_token_usage
        Config.exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        if is_openai_v1():
            OpenAIV1Instrumentor().instrument(**kwargs)
        else:
            OpenAIV0Instrumentor().instrument(**kwargs)

    def _uninstrument(self, **kwargs):
        if is_openai_v1():
            OpenAIV1Instrumentor().uninstrument(**kwargs)
        else:
            OpenAIV0Instrumentor().uninstrument(**kwargs)
