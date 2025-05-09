from typing import Optional

from opentelemetry._events import EventLogger


class Config:
    exception_logger = None
    use_legacy_attributes = True
    event_logger: Optional[EventLogger] = None
