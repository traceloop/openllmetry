import httpx
import inspect
from typing import Dict, Optional, Any, List, Callable, get_type_hints, get_origin, get_args

from .model import (
    InputExtractor,
    InputSchemaMapping,
    ExecuteEvaluatorRequest,
    ExecuteEvaluatorResponse,
    ExecutionResponse,
)
from .stream_client import SSEClient
from .config import EvaluatorDetails


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
        evaluator_config: Optional[Dict[str, Any]] = None,
    ) -> ExecuteEvaluatorRequest:
        """Build evaluator request with common parameters"""
        schema_mapping = InputSchemaMapping(
            root={k: InputExtractor(source=v) for k, v in input.items()}
        )
        return ExecuteEvaluatorRequest(
            input_schema_mapping=schema_mapping,
            evaluator_version=evaluator_version,
            evaluator_config=evaluator_config,
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
            error_detail = _extract_error_from_response(response)
            raise Exception(
                f"Failed to execute evaluator '{evaluator_slug}': "
                f"{response.status_code} - {error_detail}"
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
        evaluator_config: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResponse:
        """
        Execute evaluator with input schema mapping and wait for result

        Args:
            evaluator_slug: Slug of the evaluator to execute
            task_id: Task ID for the evaluation
            experiment_id: Experiment ID
            experiment_run_id: Experiment run ID
            input: Dict mapping evaluator input field names to their values. {field_name: value, ...}
            timeout_in_sec: Timeout in seconds for execution
            evaluator_version: Version of the evaluator to execute (optional)
            evaluator_config: Configuration for the evaluator (optional)

        Returns:
            ExecutionResponse: The evaluation result from SSE stream
        """
        request = self._build_evaluator_request(
            task_id, experiment_id, experiment_run_id, input, evaluator_version, evaluator_config
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
        evaluator_config: Optional[Dict[str, Any]] = None,
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
            evaluator_config: Configuration for the evaluator (optional)

        Returns:
            str: The execution_id that can be used to check results later
        """
        request = self._build_evaluator_request(
            task_id, experiment_id, experiment_run_id, input, evaluator_version, evaluator_config
        )

        execute_response = await self._execute_evaluator_request(
            evaluator_slug, request, 120
        )

        # Return execution_id without waiting for SSE result
        return execute_response.execution_id


def validate_task_output(
    task_output: Dict[str, Any],
    evaluators: List[EvaluatorDetails],
) -> None:
    """
    Validate that task output contains all required fields for the given evaluators.

    Args:
        task_output: The dictionary returned by the task function
        evaluators: List of EvaluatorDetails to validate against

    Raises:
        ValueError: If task output is missing required fields for any evaluator
    """
    if not evaluators:
        return

    # Collect all validation errors
    missing_fields_by_evaluator: Dict[str, set[str]] = {}

    for evaluator in evaluators:
        if not evaluator.required_input_fields:
            continue

        missing_fields = [
            field for field in evaluator.required_input_fields
            if field not in task_output
        ]

        if missing_fields:
            # Add to existing set or create new set
            if evaluator.slug not in missing_fields_by_evaluator:
                missing_fields_by_evaluator[evaluator.slug] = set()
            missing_fields_by_evaluator[evaluator.slug].update(missing_fields)

    # If there are any missing fields, raise a detailed error
    if missing_fields_by_evaluator:
        error_lines = ["Task output missing required fields for evaluators:"]

        for slug, fields in missing_fields_by_evaluator.items():
            error_lines.append(f"  - {slug} requires: {sorted(fields)}")

        error_lines.append(f"\nTask output contains: {list(task_output.keys())}")

        error_lines.append("\nHint: Update your task function to return a dictionary with the required fields.")

        raise ValueError("\n".join(error_lines))


def _extract_error_from_response(response: httpx.Response) -> str:
    """
    Extract error message from HTTP response.

    Tries to parse JSON and extract common error fields (error, message, msg).
    Falls back to response.text if JSON parsing fails.

    Args:
        response: The HTTP response object

    Returns:
        Extracted error message string
    """
    error_detail = response.text
    try:
        error_json = response.json()
        if isinstance(error_json, dict):
            if 'error' in error_json:
                error_detail = error_json['error']
            elif 'message' in error_json:
                error_detail = error_json['message']
            elif 'msg' in error_json:
                error_detail = error_json['msg']
    except Exception:
        pass  # Use response.text as fallback

    return error_detail
