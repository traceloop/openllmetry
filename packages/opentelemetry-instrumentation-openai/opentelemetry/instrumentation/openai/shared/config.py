from typing import Callable, Optional, Coroutine

from opentelemetry._events import EventLogger


class Config:
    enrich_assistant = False
    exception_logger = None
    get_common_metrics_attributes: Callable[[], dict] = lambda: {}
    upload_base64_image: Callable[[str, str, str, str], Coroutine[None, None, str]] = (
        lambda trace_id, span_id, image_name, base64_image_url: str
    )
    enable_trace_context_propagation: bool = True
    use_legacy_attributes = True
    event_logger: Optional[EventLogger] = None
