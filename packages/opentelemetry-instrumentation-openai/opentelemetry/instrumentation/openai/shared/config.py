from typing import Callable


class Config:
    enrich_token_usage = False
    enrich_assistant = False
    exception_logger = None
    get_common_metrics_attributes: Callable[[], dict] = lambda: {}
