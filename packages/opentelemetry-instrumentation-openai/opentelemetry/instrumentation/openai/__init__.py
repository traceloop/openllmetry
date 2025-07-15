from typing import Callable, Collection, Optional

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.openai.shared.config import Config
from opentelemetry.instrumentation.openai.utils import is_openai_v1, should_keep_inline_images
from typing_extensions import Coroutine

_instruments = ("openai >= 0.27.0",)

default_upload_base64_image = lambda trace_id, span_id, image_name, base64_image_url: f"data:image/{image_name.split('.')[-1]};base64,{base64_image_url}" \
    if should_keep_inline_images() \
    else lambda: ""

class OpenAIInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI's client library."""

    def __init__(
        self,
        enrich_assistant: bool = False,
        enrich_token_usage: bool = False,
        exception_logger=None,
        get_common_metrics_attributes: Callable[[], dict] = lambda: {},
        upload_base64_image: Optional[
            Callable[[str, str, str, str], Coroutine[None, None, str]]
        ] = None,
        enable_trace_context_propagation: bool = True,
        use_legacy_attributes: bool = True,
    ):
        super().__init__()
        Config.enrich_assistant = enrich_assistant
        Config.enrich_token_usage = enrich_token_usage
        Config.exception_logger = exception_logger
        Config.get_common_metrics_attributes = get_common_metrics_attributes
        # do not override if already set
        if not Config.upload_base64_image:
            Config.upload_base64_image = upload_base64_image \
                if upload_base64_image is not None \
                else default_upload_base64_image
        Config.enable_trace_context_propagation = enable_trace_context_propagation
        Config.use_legacy_attributes = use_legacy_attributes

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        if is_openai_v1():
            from opentelemetry.instrumentation.openai.v1 import OpenAIV1Instrumentor

            OpenAIV1Instrumentor().instrument(**kwargs)
        else:
            from opentelemetry.instrumentation.openai.v0 import OpenAIV0Instrumentor

            OpenAIV0Instrumentor().instrument(**kwargs)

    def _uninstrument(self, **kwargs):
        if is_openai_v1():
            from opentelemetry.instrumentation.openai.v1 import OpenAIV1Instrumentor

            OpenAIV1Instrumentor().uninstrument(**kwargs)
        else:
            from opentelemetry.instrumentation.openai.v0 import OpenAIV0Instrumentor

            OpenAIV0Instrumentor().uninstrument(**kwargs)
