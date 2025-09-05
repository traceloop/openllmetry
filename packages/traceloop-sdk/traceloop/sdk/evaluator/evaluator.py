import httpx
from typing import Dict, Optional

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

    _async_http_client: httpx.AsyncClient

    def __init__(self, async_http_client: httpx.AsyncClient):
        self._async_http_client = async_http_client

    @staticmethod
    def _build_evaluator_request(
        task_id: str,
        experiment_id: str,
        experiment_run_id: str,
        input: Dict[str, str],
        evaluator_version: Optional[str] = None,
    ) -> ExecuteEvaluatorRequest:
        """Build evaluator request with common parameters"""
        schema_mapping = InputSchemaMapping(
            root={k: InputExtractor(source=v) for k, v in input.items()}
        )
        return ExecuteEvaluatorRequest(
            input_schema_mapping=schema_mapping,
            evaluator_version=evaluator_version,
            task_id=task_id,
            experiment_id=experiment_id,
            experiment_run_id=experiment_run_id,
        )

    async def _execute_evaluator_request(
        self,
        evaluator_slug: str,
        request: ExecuteEvaluatorRequest,
        timeout_in_sec: int = 120,
    ) -> ExecuteEvaluatorResponse:
        """Execute evaluator request and return response"""
        body = request.model_dump()
        client = self._async_http_client
        full_url = f"/v2/evaluators/slug/{evaluator_slug}/execute"
        response = await client.post(
            full_url, json=body, timeout=httpx.Timeout(timeout_in_sec)
        )
        if response.status_code != 200:
            raise Exception(
                f"Failed to execute evaluator {evaluator_slug}: "
                f"{response.status_code} – {response.text}"
            )
        result_data = response.json()
        return ExecuteEvaluatorResponse(**result_data)

    async def run_experiment_evaluator(
        self,
        evaluator_slug: str,
        task_id: str,
        experiment_id: str,
        experiment_run_id: str,
        input: Dict[str, str],
        timeout_in_sec: int = 120,
        evaluator_version: Optional[str] = None,
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
        request = self._build_evaluator_request(
            task_id, experiment_id, experiment_run_id, input, evaluator_version
        )

        execute_response = await self._execute_evaluator_request(
            evaluator_slug, request, timeout_in_sec
        )
        sse_client = SSEClient(shared_client=self._async_http_client)
        sse_result = await sse_client.wait_for_result(
            execute_response.execution_id,
            execute_response.stream_url,
            timeout_in_sec,
        )
        return sse_result

    async def trigger_experiment_evaluator(
        self,
        evaluator_slug: str,
        task_id: str,
        experiment_id: str,
        experiment_run_id: str,
        input: Dict[str, str],
        evaluator_version: Optional[str] = None,
    ) -> str:
        """
        Trigger evaluator execution without waiting for result (fire-and-forget)

        Args:
            evaluator_slug: Slug of the evaluator to execute
            task_id: Task ID for the evaluation
            experiment_id: Experiment ID
            experiment_run_id: Experiment run ID
            input: Dict mapping evaluator input field names to their values
            evaluator_version: Version of the evaluator to execute (optional)
            client: Shared HTTP client for connection reuse (optional)

        Returns:
            str: The execution_id that can be used to check results later
        """
        request = self._build_evaluator_request(
            task_id, experiment_id, experiment_run_id, input, evaluator_version
        )

        execute_response = await self._execute_evaluator_request(
            evaluator_slug, request, 120
        )

        # Return execution_id without waiting for SSE result
        return execute_response.execution_id
