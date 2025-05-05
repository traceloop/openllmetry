from typing import Callable, Optional

from opentelemetry._events import EventLogger


class Config:
    enrich_token_usage = False
    exception_logger = None
    get_common_metrics_attributes: Callable[[], dict] = lambda: {}
    use_legacy_attributes = True
    event_logger: Optional[EventLogger] = None
