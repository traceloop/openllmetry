from opentelemetry.metrics import Counter, Histogram
from typing import Dict

class AzureSearchMetrics:
    """Metrics definitions for Azure Search integration"""
    
    def __init__(self):
        self.search_count = Counter(
            name="azure.search.requests",
            description="Number of search requests made",
            unit="1"
        )
        
        self.search_latency = Histogram(
            name="azure.search.latency",
            description="Search request latency",
            unit="ms"
        )
        
        self.results_count = Counter(
            name="azure.search.results.total",
            description="Total number of search results returned",
            unit="1"
        )