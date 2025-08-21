import os
import httpx
from typing import Dict, Optional

from traceloop.sdk.version import __version__
from .model import (
    InputExtractor,
    InputSchemaMapping,
    ExecuteEvaluatorRequest,
    ExecuteEvaluatorResponse,
    ExecutionResponse,
)
from .stream_client import SSEClient


class Evaluator:
    """
    Evaluator class for executing evaluators with SSE streaming
    """

    @classmethod
    def _create_async_client(cls) -> httpx.AsyncClient:
        """Create new async HTTP client"""
        api_key = os.environ.get("TRACELOOP_API_KEY", "")
        if not api_key:
            raise ValueError("TRACELOOP_API_KEY environment variable is required")

        return httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"traceloop-sdk/{__version__}",
            },
            timeout=httpx.Timeout(120.0),
        )

    @classmethod
    async def run_experiment_evaluator(
        cls,
        evaluator_slug: str,
        task_id: str,
        experiment_id: str,
        experiment_run_id: str,
        input: Dict[str, str],
        timeout_in_sec: int = 120,
        evaluator_version: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> ExecutionResponse:
        """
        Execute evaluator with input schema mapping and wait for result

        Args:
            evaluator_slug: Slug of the evaluator to execute
            input: Dict mapping evaluator input field names to their values. {field_name: value, ...}
            client: Shared HTTP client for connection reuse (optional)
            context_data: Context data to be passed to the evaluator (optional)
            evaluator_version: Version of the evaluator to execute (optional)
            timeout_in_sec: Timeout in seconds for execution

        Returns:
            ExecutionResponse: The evaluation result from SSE stream
        """
        schema_mapping = InputSchemaMapping(
            root={k: InputExtractor(source=v) for k, v in input.items()}
        )
        request = ExecuteEvaluatorRequest(
            input_schema_mapping=schema_mapping,
            evaluator_version=evaluator_version,
            task_id=task_id,
            experiment_id=experiment_id,
            experiment_run_id=experiment_run_id,
        )
        api_endpoint = os.environ.get("TRACELOOP_BASE_URL", "https://api.traceloop.com")
        body = request.model_dump()

        should_close_client = client is None
        if should_close_client:
            client = cls._create_async_client()
        try:
            full_url = f"{api_endpoint}/v2/evaluators/slug/{evaluator_slug}/execute"
            response = await client.post(
                full_url, json=body, timeout=httpx.Timeout(timeout_in_sec)
            )
            if response.status_code != 200:
                raise Exception(
                    f"Failed to execute evaluator {evaluator_slug}: "
                    f"{response.status_code} – {response.text}"
                )

            result_data = response.json()
            execute_response = ExecuteEvaluatorResponse(**result_data)

            sse_client = SSEClient(shared_client=client)
            sse_result = await sse_client.wait_for_result(
                execute_response.execution_id,
                execute_response.stream_url,
                timeout_in_sec,
            )
            return sse_result
        finally:
            if should_close_client:
                await client.aclose()
