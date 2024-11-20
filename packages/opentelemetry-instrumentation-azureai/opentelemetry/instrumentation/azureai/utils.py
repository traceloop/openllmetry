"""Utility functions for Azure AI Search instrumentation."""

import logging
from typing import Dict, Any, Optional
from azure.core.exceptions import HttpResponseError

logger = logging.getLogger(__name__)

def set_span_attribute(span, name: str, value: Any):
    """Sets a span attribute if the value is not None."""
    if value is not None:
        span.set_attribute(name, value)

def error_metrics_attributes(error: Exception) -> Dict[str, str]:
    """Extract error attributes for metrics."""
    attributes = {}
    
    if isinstance(error, HttpResponseError):
        attributes["error.type"] = "http_response_error"
        attributes["error.code"] = str(error.status_code)
    else:
        attributes["error.type"] = error.__class__.__name__
    
    return attributes

def shared_metrics_attributes(response) -> Dict[str, str]:
    """Extract common attributes for metrics from response."""
    attributes = {}
    
    if hasattr(response, "coverage"):
        attributes["search.coverage"] = str(response.coverage)
    
    if hasattr(response, "facets"):
        attributes["search.has_facets"] = "true" if response.facets else "false"
    
    return attributes

class Config:
    """Configuration for Azure Search instrumentation."""
    
    exception_logger = None
    get_common_metrics_attributes = lambda: {}

    @staticmethod
    def set_exception_logger(logger):
        """Sets the exception logger."""
        Config.exception_logger = logger

    @staticmethod
    def set_common_metrics_attributes(func):
        """Sets the function to get common metrics attributes."""
        Config.get_common_metrics_attributes = func