import os
import asyncio
import threading
from typing import Dict, Any

from traceloop.sdk.datasets.model import DatasetBaseModel
from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.version import __version__
from .model import (
    InputExtractor,
    InputSchemaMapping,
    ExecuteEvaluatorRequest,
    ExecuteEvaluatorResponse
)
from .stream_client import SSEClient


class Evaluator(DatasetBaseModel):
    """
    Evaluator class for executing evaluators with SSE streaming
    """
    _api_endpoint: str
    _api_key: str

    def __init__(self):
        self._api_endpoint = os.environ.get("TRACELOOP_BASE_URL", "https://api.traceloop.com")
        self._api_key = os.environ.get("TRACELOOP_API_KEY", "")
        
    @classmethod
    def _get_http_client_static(cls) -> HTTPClient:
        """Get HTTP client instance for static operations"""
        if not cls._api_key:
            raise ValueError("TRACELOOP_API_KEY environment variable is required")

        return HTTPClient(
            base_url=cls._api_endpoint,
            api_key=cls._api_key,
            version=__version__
        )

    @classmethod
    def run(cls, 
            evaluator_slug: str,
            input: Dict[str, str], 
            timeout_in_sec: int = 120) -> Dict[str, Any]:
        """
        Execute evaluator with input schema mapping and wait for result
        
        Args:
            evaluator_slug: Slug of the evaluator to execute
            input_schema_mapping: Dict mapping field names to source fields
            timeout: Timeout in seconds for execution
        
        Returns:
            Dict[str, Any]: The evaluation result from SSE stream
        """
      
        schema_mapping = InputSchemaMapping(root={k: InputExtractor(source=v) for k, v in input.items()})
        request = ExecuteEvaluatorRequest(input_schema_mapping=schema_mapping, source="experiments")
        
        http_client = cls._get_http_client_static()
        body = request.model_dump()
        print("body:", body)

        # Make API call to trigger evaluator
        result = http_client.post(
            f"evaluators/slug/{evaluator_slug}/execute",
            body
        )
        
        if result is None:
            raise Exception(f"Failed to execute evaluator {evaluator_slug}")
        
        response = ExecuteEvaluatorResponse(**result)
           
        return cls._wait_for_sse_result(response.stream_url, response.execution_id, timeout_in_sec)

    @classmethod
    def _wait_for_sse_result(cls, stream_url: str, execution_id: str, timeout_in_sec: int) -> Dict[str, Any]:
        """Wait synchronously for result from SSE stream - based on guardrails pattern"""
        result_container = {"result": None, "received": False, "error": None}
        sse_client = SSEClient(cls._api_key, cls._api_endpoint)     

        # Run async SSE stream in thread
        def run_sse_stream():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                result = loop.run_until_complete(
                    sse_client.wait_for_result(execution_id, stream_url, timeout_in_sec)
                )
                result_container["result"] = result
                result_container["received"] = True
            except Exception as e:
                result_container["error"] = str(e)
                result_container["received"] = True
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_sse_stream)
        thread.start()
        thread.join(timeout_in_sec + 5)  # Add small buffer to thread timeout
        
        if not result_container["received"]:
            sse_client.stop_stream(execution_id)
            raise TimeoutError(f"Evaluator execution {execution_id} timed out after {timeout}s")
        
        if result_container["error"]:
            raise Exception(f"SSE stream error: {result_container['error']}")
        
        return result_container["result"]

