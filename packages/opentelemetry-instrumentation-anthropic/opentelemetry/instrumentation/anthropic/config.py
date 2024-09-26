from typing import Callable, Optional


class Config:
    enrich_token_usage = False
    exception_logger = None
    get_common_metrics_attributes: Callable[[], dict] = lambda: {}
    upload_base64_image: Optional[Callable[[str, str, str, str], str]] = None
