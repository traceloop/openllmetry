from typing import Callable, Optional

from opentelemetry._logs import Logger
from typing_extensions import Coroutine


class Config:
    enrich_assistant = False
    exception_logger = None
    get_common_metrics_attributes: Callable[[], dict] = lambda: {}
    upload_base64_image: Optional[
        Callable[[str, str, str, str], Coroutine[None, None, str]]
    ] = None
    enable_trace_context_propagation: bool = True
    use_legacy_attributes = True
    event_logger: Optional[Logger] = None
