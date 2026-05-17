from typing import Optional

from opentelemetry._logs import Logger


class Config:
    exception_logger = None
    use_legacy_attributes = True
    event_logger: Optional[Logger] = None
