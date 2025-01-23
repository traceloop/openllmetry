"""OpenTelemetry SageMaker Instrumentation Configuration"""

class Config:
    """Configuration for the SageMaker instrumentation."""

    def __init__(
        self,
        use_legacy_attributes: bool = True,
        capture_content: bool = True,
        exception_logger=None,
        enrich_token_usage: bool = False,
    ):
        """Initialize configuration.

        Args:
            use_legacy_attributes: Whether to use legacy attribute-based approach (default: True)
            capture_content: Whether to capture prompt and completion content (default: True)
            exception_logger: Optional exception logger
            enrich_token_usage: Whether to enrich spans with token usage metrics (default: False)
        """
        self.use_legacy_attributes = use_legacy_attributes
        self.capture_content = capture_content
        self.exception_logger = exception_logger
        self.enrich_token_usage = enrich_token_usage
