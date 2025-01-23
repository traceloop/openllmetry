from typing import Callable, Optional
from typing_extensions import Coroutine


class Config:
    # Legacy attribute-based tracking (default: True for backward compatibility)
    use_legacy_attributes = True

    # Token usage enrichment
    enrich_token_usage = False
    
    # Event logging configuration
    capture_content = True  # Whether to capture prompt/completion content in events
    
    # Error handling
    exception_logger = None
    
    # Metrics and attributes
    get_common_metrics_attributes: Callable[[], dict] = lambda: {}
    
    # Image handling
    upload_base64_image: Optional[
        Callable[[str, str, str, str], Coroutine[None, None, str]]
    ] = None
