class Config:
    """Configuration for Bedrock instrumentation."""

    enrich_token_usage = False
    exception_logger = None
    use_legacy_attributes: bool = True  # Controls whether to use legacy attributes or new event-based semantic conventions
