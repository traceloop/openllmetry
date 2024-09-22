from typing import Callable, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

from opentelemetry.instrumentation.openai.shared.config import Config
from opentelemetry.instrumentation.openai.utils import is_openai_v1
from opentelemetry.instrumentation.openai.v0 import OpenAIV0Instrumentor
from opentelemetry.instrumentation.openai.v1 import OpenAIV1Instrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

_instruments = ("openai >= 0.27.0",)


class OpenAIInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI's client library."""

    def __init__(
        self,
        enrich_assistant: bool = False,
        enrich_token_usage: bool = False,
        exception_logger=None,
        get_common_metrics_attributes: Callable[[], dict] = lambda: {},
        upload_base64_image: Callable[[str, str, str, str], str] = lambda trace_id, span_id, image_name, base64_image_url: str,
    ):
        super().__init__()
        Config.enrich_assistant = enrich_assistant
        Config.enrich_token_usage = enrich_token_usage
        Config.exception_logger = exception_logger
        Config.get_common_metrics_attributes = get_common_metrics_attributes
        Config.upload_base64_image = upload_base64_image
        self._requests_instrumentor = RequestsInstrumentor()


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
