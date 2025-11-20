class Config:
    """Global configuration for Agno instrumentation."""

    exception_logger = None
    """Optional logger for recording instrumentation exceptions."""

    enrich_assistant = True
    """Whether to enrich spans with assistant information."""

    enrich_token_usage = False
    """Whether to enrich spans with detailed token usage information."""

    use_legacy_attributes = True
    """Whether to use legacy span attribute names for backward compatibility."""
