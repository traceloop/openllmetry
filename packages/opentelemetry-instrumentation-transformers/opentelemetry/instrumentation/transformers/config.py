"""OpenTelemetry Transformers Instrumentation Configuration"""

class Config:
    """Configuration for the Transformers instrumentation."""

    def __init__(
        self,
        use_legacy_attributes: bool = True,
        capture_content: bool = True,
        exception_logger=None,
    ):
        """Initialize configuration.

        Args:
            use_legacy_attributes: Whether to use legacy attribute-based approach (default: True)
            capture_content: Whether to capture prompt and completion content (default: True)
            exception_logger: Optional exception logger
        """
        self.use_legacy_attributes = use_legacy_attributes
        self.capture_content = capture_content
        self.exception_logger = exception_logger
