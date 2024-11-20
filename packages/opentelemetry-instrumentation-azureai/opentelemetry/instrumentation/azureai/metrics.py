"""Metrics utilities for Azure Search OpenTelemetry instrumentation."""

from typing import Dict, Optional, Any
from opentelemetry.metrics import Meter, Counter, Histogram
from opentelemetry.semconv_ai import Meters

class SearchMetricsRecorder:
    """Class to handle recording of Azure Search metrics."""
    
    def __init__(self, meter: Meter):
        """Initialize metrics recorder with OpenTelemetry meter.
        
        Args:
            meter: OpenTelemetry meter instance
        """
        self._duration_histogram = self._create_duration_histogram(meter)
        self._result_counter = self._create_result_counter(meter)
        self._exception_counter = self._create_exception_counter(meter)

    @staticmethod
    def _create_duration_histogram(meter: Meter) -> Histogram:
        """Create histogram for operation duration tracking.
        
        Args:
            meter: OpenTelemetry meter instance
            
        Returns:
            Histogram metric for tracking operation duration
        """
        return meter.create_histogram(
            name=Meters.LLM_OPERATION_DURATION,
            unit="s",
            description="Azure AI Search operation duration"
        )

    @staticmethod
    def _create_result_counter(meter: Meter) -> Counter:
        """Create counter for tracking search results.
        
        Args:
            meter: OpenTelemetry meter instance
            
        Returns:
            Counter metric for tracking search results
        """
        return meter.create_counter(
            name=f"{Meters.LLM_AZURE_SEARCH_PREFIX}.results",
            unit="result",
            description="Number of results returned by search operation"
        )

    @staticmethod
    def _create_exception_counter(meter: Meter) -> Counter:
        """Create counter for tracking exceptions.
        
        Args:
            meter: OpenTelemetry meter instance
            
        Returns:
            Counter metric for tracking exceptions
        """
        return meter.create_counter(
            name=f"{Meters.LLM_AZURE_SEARCH_PREFIX}.exceptions",
            unit="exception",
            description="Number of exceptions occurred during search operations"
        )

    def record_duration(self, duration: float, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Record operation duration.
        
        Args:
            duration: Operation duration in seconds
            attributes: Optional attributes to attach to the metric
        """
        if self._duration_histogram:
            self._duration_histogram.record(duration, attributes=attributes or {})

    def record_results(self, count: int, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Record number of search results.
        
        Args:
            count: Number of results returned
            attributes: Optional attributes to attach to the metric
        """
        if self._result_counter and count is not None:
            self._result_counter.add(count, attributes=attributes or {})

    def record_exception(self, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Record occurrence of an exception.
        
        Args:
            attributes: Optional attributes to attach to the metric
        """
        if self._exception_counter:
            self._exception_counter.add(1, attributes=attributes or {})

def create_metrics_recorder(meter: Meter) -> SearchMetricsRecorder:
    """Create a new metrics recorder instance.
    
    Args:
        meter: OpenTelemetry meter instance
        
    Returns:
        SearchMetricsRecorder instance
    """
    return SearchMetricsRecorder(meter)