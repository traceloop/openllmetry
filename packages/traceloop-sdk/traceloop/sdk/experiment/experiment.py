import os
import uuid
from datetime import datetime
from typing import Dict, Any

from .experiment_context import ExperimentContext   
from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.version import __version__

class Experiment(): 
    """Main Experiment class for creating experiment contexts"""
    id: str
    name: str
    created_at: datetime
    run_data: Dict[str, Any]
 
    def __init__(self, name: str, run_data: Dict[str, Any]):
        self.name = name
        self.created_at = datetime.now()
        self.id = str(uuid.uuid4())
        self.run_data = run_data
        self._http_client = self._get_http_client()

    def _get_http_client(self) -> HTTPClient:
        api_key = os.getenv("TRACELOOP_API_KEY")
        if not api_key:
            raise Exception("TRACELOOP_API_KEY is not set")
        api_endpoint = os.getenv("TRACELOOP_BASE_URL", "https://api.traceloop.com")
        return HTTPClient(
            base_url=api_endpoint, api_key=api_key, version=__version__
        )
    
    def run(self) -> ExperimentContext:
        """Create a new experiment context"""
        return ExperimentContext(self)
    
    