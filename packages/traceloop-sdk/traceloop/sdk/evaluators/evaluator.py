import os
import httpx
from typing import Dict, Any

from traceloop.sdk.version import __version__
from .model import (
    InputExtractor,
    InputSchemaMapping,
    ExecuteEvaluatorRequest,
    ExecuteEvaluatorResponse
)
from .stream_client import SSEClient


class Evaluator:
    """
    Evaluator class for executing evaluators with SSE streaming
    """
        

    @classmethod
    async def run(cls, 
                  evaluator_slug: str,
                  input: Dict[str, str], 
                  client: httpx.AsyncClient,
                  timeout_in_sec: int = 120) -> Dict[str, Any]:
        """
        Execute evaluator with input schema mapping and wait for result
        
        Args:
            evaluator_slug: Slug of the evaluator to execute
            input: Dict mapping evaluator input field names to their values. {field_name: value, ...}
            client: Shared HTTP client for connection reuse
            timeout_in_sec: Timeout in seconds for execution
        
        Returns:
            Dict[str, Any]: The evaluation result from SSE stream
        """
        schema_mapping = InputSchemaMapping(root={k: InputExtractor(source=v) for k, v in input.items()})
        request = ExecuteEvaluatorRequest(input_schema_mapping=schema_mapping, source="experiments")
        
        api_endpoint = os.environ.get("TRACELOOP_BASE_URL", "https://api.traceloop.com")
        body = request.model_dump()
        
        # Make API call to trigger evaluator
        response = await client.post(
            f"{api_endpoint}/evaluators/slug/{evaluator_slug}/execute",
            json=body,
            timeout=timeout_in_sec
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to execute evaluator {evaluator_slug}: {response.status_code}")
        
        result_data = response.json()
        execute_response = ExecuteEvaluatorResponse(**result_data)
        
        # Wait for SSE result using async SSE client with shared HTTP client
        sse_client = SSEClient(shared_client=client)
        return await sse_client.wait_for_result(
            execute_response.execution_id,
            execute_response.stream_url, 
            timeout_in_sec
        )


