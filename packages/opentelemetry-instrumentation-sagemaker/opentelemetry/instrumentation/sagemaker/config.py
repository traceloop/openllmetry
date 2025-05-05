from typing import Optional

from opentelemetry._events import EventLogger


class Config:
    enrich_token_usage = False
    exception_logger = None
    use_legacy_attributes = True
    event_logger: Optional[EventLogger] = None
