from typing import Callable


class Config:
    exception_logger = None
    get_common_metrics_attributes: Callable[[], dict] = lambda: {}
    use_legacy_attributes = True
