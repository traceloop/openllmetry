import os
import asyncio
import threading
from typing import Dict, Any, Optional, Callable, Union

from traceloop.sdk.datasets.model import DatasetBaseModel
from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.version import __version__
from .model import (
    InputExtractor,
    InputSchemaMapping,
    ExecuteEvaluatorRequest,
    ExecuteEvaluatorResponse
)
from .stream_client import SSEResultClient


class Evaluator(DatasetBaseModel):
    """
    Evaluator class for executing evaluators with SSE streaming
    """
    
    @classmethod
    def _get_http_client_static(cls) -> HTTPClient:
        """Get HTTP client instance for static operations"""
        api_key = os.environ.get("TRACELOOP_API_KEY", "")
        api_endpoint = os.environ.get("TRACELOOP_BASE_URL", "https://api.traceloop.com")

        if not api_key:
            raise ValueError("TRACELOOP_API_KEY environment variable is required")

        return HTTPClient(
            base_url=api_endpoint,
            api_key=api_key,
            version=__version__
        )

    @classmethod
    def run(cls, 
            evaluator_slug: str,
            input_schema_mapping: Dict[str, str], 
            callback: Optional[Callable[[Dict[str, Any]], None]] = None,
            wait_for_result: bool = True,
            timeout: int = 500) -> Union[ExecuteEvaluatorResponse, Dict[str, Any]]:
        """
        Execute evaluator with input schema mapping
        
        Args:
            evaluator_slug: Slug of the evaluator to execute
            input_schema_mapping: Dict mapping field names to source fields
            callback: Optional callback function for async result handling
            wait_for_result: If True, blocks until result is received via SSE stream
            timeout: Timeout in seconds for synchronous execution
        
        Returns:
            ExecuteEvaluatorResponse (immediate) or Dict[str, Any] (if waiting for result)
        """
        # Convert dict to proper model format
        mapping = InputSchemaMapping(__root__={
            field: InputExtractor(source=source) 
            for field, source in input_schema_mapping.items()
        })
        
        request = ExecuteEvaluatorRequest(input_schema_mapping=mapping)
        
        # Get HTTP client
        http_client = cls._get_http_client_static()
        
        # Make API call to trigger evaluator
        result = http_client.post(
            f"v2/evaluators/slug/{evaluator_slug}/execute",
            request.model_dump(by_alias=True)
        )
        
        if result is None:
            raise Exception(f"Failed to execute evaluator {evaluator_slug}")
        
        response = ExecuteEvaluatorResponse(**result)
        
        # Handle SSE streaming results
        if callback or wait_for_result:
            api_key = os.environ.get("TRACELOOP_API_KEY", "")
            sse_client = SSEResultClient(api_key)
            
            if wait_for_result:
                # Synchronous execution - wait for result
                return cls._wait_for_sse_result(sse_client, response.stream_url, response.execution_id, timeout)
            else:
                # Asynchronous execution - start stream with callback
                sse_client.start_async_stream(response.execution_id, response.stream_url, callback, timeout)
        
        return response

    @classmethod
    def _wait_for_sse_result(cls, sse_client: SSEResultClient, stream_url: str, execution_id: str, timeout: int) -> Dict[str, Any]:
        """Wait synchronously for result from SSE stream - based on guardrails pattern"""
        result_container = {"result": None, "received": False, "error": None}
        
        # Run async SSE stream in thread
        def run_sse_stream():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    sse_client.wait_for_result(execution_id, stream_url, timeout)
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
        thread.join(timeout + 5)  # Add small buffer to thread timeout
        
        if not result_container["received"]:
            sse_client.stop_stream(execution_id)
            raise TimeoutError(f"Evaluator execution {execution_id} timed out after {timeout}s")
        
        if result_container["error"]:
            raise Exception(f"SSE stream error: {result_container['error']}")
        
        return result_container["result"]

    @classmethod
    async def run_async(cls, 
                       evaluator_slug: str,
                       input_schema_mapping: Dict[str, str],
                       callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                       timeout: int = 120) -> ExecuteEvaluatorResponse:
        """Async version of run method"""
        # Convert dict to proper model format
        mapping = InputSchemaMapping(__root__={
            field: InputExtractor(source=source) 
            for field, source in input_schema_mapping.items()
        })
        
        request = ExecuteEvaluatorRequest(input_schema_mapping=mapping)
        
        # Get HTTP client
        http_client = cls._get_http_client_static()
        
        # Make API call to trigger evaluator
        result = http_client.post(
            f"v2/evaluators/slug/{evaluator_slug}/execute",
            request.model_dump(by_alias=True)
        )
        
        if result is None:
            raise Exception(f"Failed to execute evaluator {evaluator_slug}")
        
        response = ExecuteEvaluatorResponse(**result)
        
        # Set up SSE client for result delivery if callback provided
        if callback:
            api_key = os.environ.get("TRACELOOP_API_KEY", "")
            sse_client = SSEResultClient(api_key)
            sse_client.start_async_stream(response.execution_id, response.stream_url, callback, timeout)
        
        return response