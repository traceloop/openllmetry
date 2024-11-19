from typing import Dict, List, Optional, Any
from opentelemetry.base import BaseIntegration
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

class AzureSearchIntegration(BaseIntegration):
    """Integration for Azure AI Search in OpenLLMetry"""
    
    def __init__(
        self,
        endpoint: str,
        key: str,
        index_name: str,
        service_name: str = "azure-search"
    ):
        super().__init__(service_name)
        self.endpoint = endpoint
        self.credential = AzureKeyCredential(key)
        self.index_name = index_name
        self.client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=self.credential
        )
        
    def search(
        self,
        query: str,
        filter: Optional[str] = None,
        top: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform a search operation and collect metrics
        """
        with self.start_span("azure.search.query") as span:
            # Add relevant attributes to span
            span.set_attribute("azure.search.query", query)
            span.set_attribute("azure.search.index", self.index_name)
            if filter:
                span.set_attribute("azure.search.filter", filter)
            if top:
                span.set_attribute("azure.search.top", top)
                
            try:
                results = list(self.client.search(
                    search_text=query,
                    filter=filter,
                    top=top,
                    **kwargs
                ))
                
                # Record metrics
                self.record_metric(
                    "azure.search.results.count",
                    len(results),
                    {"index": self.index_name}
                )
                
                return {
                    "results": results,
                    "count": len(results)
                }
                
            except Exception as e:
                # Record error in span
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise