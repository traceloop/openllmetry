class Config:
    """Configuration for Langchain instrumentation."""
    
    def __init__(
        self,
        use_legacy_attributes: bool = True,
        exception_logger=None,
        capture_content: bool = True,
        disable_trace_context_propagation: bool = False,
    ):
        """Initialize configuration.
        
        Args:
            use_legacy_attributes: Whether to use legacy attribute-based approach (default: True)
            exception_logger: Optional logger for exceptions
            capture_content: Whether to capture prompt and completion content (default: True)
            disable_trace_context_propagation: Whether to disable trace context propagation (default: False)
        """
        self.use_legacy_attributes = use_legacy_attributes
        self.exception_logger = exception_logger
        self.capture_content = capture_content
        self.disable_trace_context_propagation = disable_trace_context_propagation
