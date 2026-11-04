from typing import Callable


class Config:
    exception_logger = None
    use_legacy_attributes = True
    get_common_metrics_attributes: Callable[[], dict] = lambda: {}
