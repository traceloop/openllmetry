import cuid
import asyncio
import json
from typing import Any, List, Callable, Optional, Tuple, Dict
from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.datasets.datasets import Datasets
from traceloop.sdk.evaluator.evaluator import Evaluator
from traceloop.sdk.experiment.model import (
    InitExperimentRequest,
    ExperimentInitResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    EvaluatorDetails,
    TaskResponse,
)
import httpx


class Experiment:
    """Main Experiment class for creating experiment contexts"""

    _datasets: Datasets
    _evaluator: Evaluator
    _http_client: HTTPClient

    def __init__(self, http_client: HTTPClient, async_http_client: httpx.AsyncClient, experiment_slug: str):
        self._datasets = Datasets(http_client)
        self._evaluator = Evaluator(async_http_client)
        self._http_client = http_client
        self._experiment_slug = experiment_slug

    async def run(
        self,
        task: Callable[[Optional[Dict[str, Any]]], Dict[str, Any]],
        dataset_slug: Optional[str] = None,
        dataset_version: Optional[str] = None,
        evaluators: Optional[List[EvaluatorDetails]] = None,
        experiment_slug: Optional[str] = None,
        related_ref: Optional[Dict[str, str]] = None,
        aux: Optional[Dict[str, str]] = None,
        stop_on_error: bool = False,
        wait_for_results: bool = True,
    ) -> Tuple[List[TaskResponse], List[str]]:
        """Run an experiment with the given task and evaluators

        Args:
            dataset_slug: Slug of the dataset to use
            task: Function to run on each dataset row
            evaluators: List of evaluator slugs to run
            experiment_slug: Slug for this experiment run
            related_ref: Related reference for this experiment run
            aux: Auxiliary information for this experiment run
            stop_on_error: Whether to stop on first error (default: False)
            wait_for_results: Whether to wait for async tasks to complete (default: True)

        Returns:
            Tuple of (results, errors). Returns ([], []) if wait_for_results is False
        """

        if not experiment_slug:
            experiment_slug = self._experiment_slug or "exp-" + str(cuid.cuid())[:11]

        experiment_run_metadata = {
            key: value
            for key, value in [("related_ref", related_ref), ("aux", aux)]
            if value is not None
        }

        evaluator_details = (
            [
                (evaluator, None) if isinstance(evaluator, str) else evaluator
                for evaluator in evaluators
            ]
            if evaluators
            else None
        )

        experiment = self._init_experiment(
            experiment_slug,
            dataset_slug=dataset_slug,
            dataset_version=dataset_version,
            evaluator_slugs=[slug for slug, _ in evaluator_details]
            if evaluator_details
            else None,
            experiment_run_metadata=experiment_run_metadata,
        )

        run_id = experiment.run.id

        rows = []
        if dataset_slug and dataset_version:
            jsonl_data = self._datasets.get_version_jsonl(dataset_slug, dataset_version)
            rows = self._parse_jsonl_to_rows(jsonl_data)

        results: List[TaskResponse] = []
        errors: List[str] = []

        async def run_single_row(row) -> TaskResponse:
            try:
                task_result = await task(row)
                task_id = self._create_task(
                    experiment_slug=experiment_slug,
                    experiment_run_id=run_id,
                    task_input=row,
                    task_output=task_result,
                ).id

                eval_results = {}
                if evaluator_details:
                    for evaluator_slug, evaluator_version in evaluator_details:
                        try:
                            if wait_for_results:
                                eval_result = (
                                    await self._evaluator.run_experiment_evaluator(
                                        evaluator_slug=evaluator_slug,
                                        evaluator_version=evaluator_version,
                                        task_id=task_id,
                                        experiment_id=experiment.experiment.id,
                                        experiment_run_id=run_id,
                                        input=task_result,
                                        timeout_in_sec=120,
                                    )
                                )
                                eval_results[evaluator_slug] = eval_result.result
                            else:
                                await self._evaluator.trigger_experiment_evaluator(
                                    evaluator_slug=evaluator_slug,
                                    evaluator_version=evaluator_version,
                                    task_id=task_id,
                                    experiment_id=experiment.experiment.id,
                                    experiment_run_id=run_id,
                                    input=task_result,
                                )

                                eval_results[evaluator_slug] = (
                                    f"Triggered execution of {evaluator_slug}"
                                )

                        except Exception as e:
                            eval_results[evaluator_slug] = f"Error: {str(e)}"

                return TaskResponse(
                    task_result=task_result,
                    evaluations=eval_results,
                )
            except Exception as e:
                error_msg = f"Error processing row: {str(e)}"
                if stop_on_error:
                    raise e
                return TaskResponse(error=error_msg)

        semaphore = asyncio.Semaphore(50)

        async def run_with_semaphore(row) -> TaskResponse:
            async with semaphore:
                return await run_single_row(row)

        tasks = [asyncio.create_task(run_with_semaphore(row)) for row in rows]

        if not wait_for_results:
            # Still need to execute tasks to trigger evaluators, but don't wait for completion
            await asyncio.gather(*tasks, return_exceptions=True)
            return [], []

        for completed_task in asyncio.as_completed(tasks):
            try:
                result = await completed_task
                if result.error:
                    errors.append(result.error)
                else:
                    results.append(result)
            except Exception as e:
                error_msg = f"Task execution error: {str(e)}"
                errors.append(error_msg)
                if stop_on_error:
                    break

        return results, errors

    def _init_experiment(
        self,
        experiment_slug: str,
        dataset_slug: Optional[str] = None,
        dataset_version: Optional[str] = None,
        evaluator_slugs: Optional[List[str]] = None,
        experiment_metadata: Optional[Dict[str, Any]] = None,
        experiment_run_metadata: Optional[Dict[str, Any]] = None,
    ) -> ExperimentInitResponse:
        """Get experiment by slug from API"""
        body = InitExperimentRequest(
            slug=experiment_slug,
            dataset_slug=dataset_slug,
            dataset_version=dataset_version,
            evaluator_slugs=evaluator_slugs,
            experiment_metadata=experiment_metadata,
            experiment_run_metadata=experiment_run_metadata,
        )
        response = self._http_client.put(
            "/experiments/initialize", body.model_dump(mode="json")
        )
        if response is None:
            raise Exception(
                f"Failed to create or fetch experiment with slug '{experiment_slug}'"
            )
        return ExperimentInitResponse(**response)

    def _create_task(
        self,
        experiment_slug: str,
        experiment_run_id: str,
        task_input: Dict[str, Any],
        task_output: Dict[str, Any],
    ) -> CreateTaskResponse:
        body = CreateTaskRequest(
            input=task_input,
            output=task_output,
        )
        response = self._http_client.post(
            f"/experiments/{experiment_slug}/runs/{experiment_run_id}/task",
            body.model_dump(mode="json"),
        )
        if response is None:
            raise Exception(f"Failed to create task for experiment '{experiment_slug}'")
        return CreateTaskResponse(**response)

    def _parse_jsonl_to_rows(self, jsonl_data: str) -> List[Dict[str, Any]]:
        """Parse JSONL string into list of {col_name: col_value} dictionaries"""
        rows = []
        lines = jsonl_data.strip().split("\n")

        # Skip the first line (columns definition)
        for line in lines[1:]:
            if line.strip():
                try:
                    row_data = json.loads(line)
                    rows.append(row_data)
                except json.JSONDecodeError:
                    # Skip invalid JSON lines
                    continue

        return rows
