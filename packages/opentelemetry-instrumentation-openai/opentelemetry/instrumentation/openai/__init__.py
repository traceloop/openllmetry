import warnings
from typing import Callable, Collection, Optional

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.openai.shared.config import Config
from opentelemetry.instrumentation.openai.utils import is_openai_v1
from typing_extensions import Coroutine

_instruments = ("openai >= 0.27.0",)


class OpenAIInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI's client library."""

    def __init__(
        self,
        enrich_assistant: bool = False,
        exception_logger=None,
        get_common_metrics_attributes: Callable[[], dict] = lambda: {},
        upload_base64_image: Optional[
            Callable[[str, str, str, str], Coroutine[None, None, str]]
        ] = lambda *args: "",
        enable_trace_context_propagation: bool = True,
        use_attributes: Optional[bool] = None,
        use_legacy_attributes: Optional[bool] = None,
    ):
        super().__init__()
        if use_attributes is not None and use_legacy_attributes is not None:
            raise TypeError(
                "Cannot pass both `use_attributes` and `use_legacy_attributes`; "
                "`use_legacy_attributes` is deprecated, use `use_attributes` instead."
            )
        if use_legacy_attributes is not None:
            warnings.warn(
                "`use_legacy_attributes` is deprecated and will be removed in a "
                "future release; use `use_attributes` instead. The current OTel "
                "GenAI spec emits prompts/completions as span attributes "
                "(`gen_ai.input.messages` / `gen_ai.output.messages`), which is "
                "what `use_attributes=True` (the default) does. "
                "`use_attributes=False` opts into the events path instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            use_attributes = use_legacy_attributes
        if use_attributes is None:
            use_attributes = True
        Config.enrich_assistant = enrich_assistant
        Config.exception_logger = exception_logger
        Config.get_common_metrics_attributes = get_common_metrics_attributes
        Config.upload_base64_image = upload_base64_image
        Config.enable_trace_context_propagation = enable_trace_context_propagation
        Config.use_legacy_attributes = use_attributes

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
