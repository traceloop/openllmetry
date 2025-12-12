import cuid
import asyncio
import json
import os
from typing import Any, List, Callable, Optional, Tuple, Dict, Awaitable, Union
from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.datasets.datasets import Datasets
from traceloop.sdk.evaluator.evaluator import Evaluator, validate_and_normalize_task_output
from traceloop.sdk.experiment.model import (
    InitExperimentRequest,
    ExperimentInitResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    EvaluatorSpec,
    TaskResponse,
    RunInGithubRequest,
    RunInGithubResponse,
    TaskResult,
    GithubContext,
)
from traceloop.sdk.evaluator.config import EvaluatorDetails
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
        task: Callable[[Optional[Dict[str, Any]]], Awaitable[Dict[str, Any]]],
        evaluators: List[EvaluatorSpec],
        dataset_slug: Optional[str] = None,
        dataset_version: Optional[str] = None,
        experiment_slug: Optional[str] = None,
        experiment_metadata: Optional[Dict[str, Any]] = None,
        related_ref: Optional[Dict[str, str]] = None,
        aux: Optional[Dict[str, str]] = None,
        stop_on_error: bool = False,
        wait_for_results: bool = True,
    ) -> Tuple[List[TaskResponse], List[str]] | RunInGithubResponse:
        """Run an experiment with the given task and evaluators

        Args:
            task: Async function to run on each dataset row
            evaluators: List of evaluator slugs or EvaluatorDetails objects to run
            dataset_slug: Slug of the dataset to use
            dataset_version: Version of the dataset to use
            experiment_slug: Slug for this experiment run
            experiment_metadata: Metadata for this experiment (an experiment holds all the experiment runs)
            related_ref: Related reference for this experiment run
            aux: Auxiliary information for this experiment run
            stop_on_error: Whether to stop on first error (default: False)
            wait_for_results: Whether to wait for async tasks to complete (default: True)
        Returns:
            Tuple of (results, errors). Returns ([], []) if wait_for_results is False
        """
        if os.getenv("GITHUB_ACTIONS"):
            return await self._run_in_github(
                task=task,
                dataset_slug=dataset_slug,
                dataset_version=dataset_version,
                evaluators=evaluators,
                experiment_slug=experiment_slug,
                related_ref=related_ref,
                aux=aux,
            )
        else:
            return await self._run_locally(
                task=task,
                evaluators=evaluators,
                dataset_slug=dataset_slug,
                dataset_version=dataset_version,
                experiment_slug=experiment_slug,
                experiment_metadata=experiment_metadata,
                related_ref=related_ref,
                aux=aux,
                stop_on_error=stop_on_error,
                wait_for_results=wait_for_results,
            )

    async def _run_locally(
        self,
        task: Callable[[Optional[Dict[str, Any]]], Awaitable[Dict[str, Any]]],
        evaluators: List[EvaluatorSpec],
        dataset_slug: Optional[str] = None,
        dataset_version: Optional[str] = None,
        experiment_slug: Optional[str] = None,
        experiment_metadata: Optional[Dict[str, Any]] = None,
        related_ref: Optional[Dict[str, str]] = None,
        aux: Optional[Dict[str, str]] = None,
        stop_on_error: bool = False,
        wait_for_results: bool = True,
    ) -> Tuple[List[TaskResponse], List[str]]:
        """Run an experiment with the given task and evaluators

        Args:
            dataset_slug: Slug of the dataset to use
            task: Async function to run on each dataset row
            evaluators: List of evaluator slugs to run
            experiment_slug: Slug for this experiment run
            experiment_metadata: Metadata for this experiment (an experiment holds all the experiment runs)
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

        # Convert evaluators to tuples of (slug, version, config)
        evaluator_details: Optional[List[Tuple[str, Optional[str], Optional[Dict[str, Any]]]]] = None
        if evaluators:
            evaluator_details = []
            for evaluator in evaluators:
                if isinstance(evaluator, str):
                    # Simple string slug
                    evaluator_details.append((evaluator, None, None))
                elif isinstance(evaluator, EvaluatorDetails):
                    # EvaluatorDetails object with config
                    evaluator_details.append((evaluator.slug, evaluator.version, evaluator.config))

        experiment = self._init_experiment(
            experiment_slug,
            dataset_slug=dataset_slug,
            dataset_version=dataset_version,
            evaluator_slugs=[slug for slug, _, _ in evaluator_details]
            if evaluator_details
            else None,
            experiment_metadata=experiment_metadata,
            experiment_run_metadata=experiment_run_metadata,
        )

        run_id = experiment.run.id

        rows = []
        if dataset_slug and dataset_version:
            jsonl_data = self._datasets.get_version_jsonl(dataset_slug, dataset_version)
            rows = self._parse_jsonl_to_rows(jsonl_data)

        results: List[TaskResponse] = []
        errors: List[str] = []

        evaluators_to_validate = [evaluator for evaluator in evaluators if isinstance(evaluator, EvaluatorDetails)]

        async def run_single_row(row: Optional[Dict[str, Any]]) -> TaskResponse:
            try:
                task_result = await task(row)

                # Validate task output with EvaluatorDetails and normalize field names using synonyms
                if evaluators_to_validate:
                    task_result = validate_and_normalize_task_output(task_result, evaluators_to_validate)

                task_id = self._create_task(
                    experiment_slug=experiment_slug,
                    experiment_run_id=run_id,
                    task_input=row,
                    task_output=task_result,
                ).id

                eval_results: Dict[str, Union[Dict[str, Any], str]] = {}
                if evaluator_details:
                    for evaluator_slug, evaluator_version, evaluator_config in evaluator_details:
                        try:
                            if wait_for_results:
                                eval_result = (
                                    await self._evaluator.run_experiment_evaluator(
                                        evaluator_slug=evaluator_slug,
                                        evaluator_version=evaluator_version,
                                        evaluator_config=evaluator_config,
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
                                    evaluator_config=evaluator_config,
                                    task_id=task_id,
                                    experiment_id=experiment.experiment.id,
                                    experiment_run_id=run_id,
                                    input=task_result,
                                )

                                msg = f"Triggered execution of {evaluator_slug}"
                                eval_results[evaluator_slug] = msg

                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            eval_results[evaluator_slug] = error_msg
                            # Log the error so user can see it
                            print(f"\033[91m❌ Evaluator '{evaluator_slug}' failed: {str(e)}\033[0m")

                return TaskResponse(
                    task_result=task_result,
                    evaluations=eval_results,
                )
            except Exception as e:
                error_msg = f"Error processing row: {str(e)}"
                # Print error to console so user can see it
                print(f"\033[91m❌ Task execution failed: {str(e)}\033[0m")
                if stop_on_error:
                    raise e
                return TaskResponse(error=error_msg)

        semaphore = asyncio.Semaphore(50)

        async def run_with_semaphore(row: Optional[Dict[str, Any]]) -> TaskResponse:
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

    async def _run_in_github(
        self,
        task: Callable[[Optional[Dict[str, Any]]], Awaitable[Dict[str, Any]]],
        evaluators: List[EvaluatorSpec],
        dataset_slug: Optional[str] = None,
        dataset_version: Optional[str] = None,
        experiment_slug: Optional[str] = None,
        experiment_metadata: Optional[Dict[str, Any]] = None,
        related_ref: Optional[Dict[str, str]] = None,
        aux: Optional[Dict[str, str]] = None,
    ) -> RunInGithubResponse:
        """Execute tasks locally and submit results to backend for GitHub CI/CD

        This method:
        1. Fetches the dataset
        2. Executes all tasks locally
        3. Sends task results to backend
        4. Backend runs evaluators and posts PR comment

        Args:
            task: Async function to run on each dataset row
            dataset_slug: Slug of the dataset to use
            dataset_version: Version of the dataset
            evaluators: List of evaluator slugs or (slug, version) tuples to run
            experiment_slug: Slug for this experiment run
            experiment_metadata: Metadata for this experiment (an experiment holds all the experiment runs)
            related_ref: Additional reference information for this experiment run
            aux: Auxiliary information for this experiment run

        Returns:
            RunInGithubResponse with experiment_id, run_id, and status

        Raises:
            RuntimeError: If not running in GitHub Actions environment
            Exception: If the API request fails
        """

        # Check if running in GitHub Actions
        if not os.getenv("GITHUB_ACTIONS"):
            raise RuntimeError(
                "run_in_github() can only be used in GitHub Actions CI/CD environment. "
                "To run experiments locally, use the run() method instead."
            )

        if not experiment_slug:
            experiment_slug = self._experiment_slug or "exp-" + str(cuid.cuid())[:11]

        # Fetch dataset rows
        rows = []
        if dataset_slug and dataset_version:
            jsonl_data = self._datasets.get_version_jsonl(dataset_slug, dataset_version)
            rows = self._parse_jsonl_to_rows(jsonl_data)

        task_results = await self._execute_tasks(rows, task, evaluators)

        # Construct GitHub context
        repository = os.getenv("GITHUB_REPOSITORY")
        server_url = os.getenv("GITHUB_SERVER_URL", "https://github.com")
        github_event_name = os.getenv("GITHUB_EVENT_NAME", "")

        # Verify this is running in a pull request context
        if github_event_name != "pull_request":
            raise RuntimeError(
                f"run_in_github() can only be used in pull_request workflow. "
                f"Current event: {github_event_name}. "
                "To run experiments locally, use the run() method instead."
            )

        # Extract PR number from GITHUB_REF (format: "refs/pull/123/merge")
        github_ref = os.getenv("GITHUB_REF", "")
        pr_number = None
        if github_ref.startswith("refs/pull/"):
            pr_number = github_ref.split("/")[2]

        if not repository or not github_ref or not pr_number:
            raise RuntimeError(
                "GITHUB_REPOSITORY and GITHUB_REF must be set in the environment. "
                "To run experiments locally, use the run() method instead."
            )

        pr_url = f"{server_url}/{repository}/pull/{pr_number}"

        github_context = GithubContext(
            repository=repository,
            pr_url=pr_url,
            commit_hash=os.getenv("GITHUB_SHA", ""),
            actor=os.getenv("GITHUB_ACTOR", ""),
        )

        experiment_metadata = dict(
            experiment_metadata or {},
            created_from="github"
        )

        experiment_run_metadata = {
            key: value
            for key, value in [("related_ref", related_ref), ("aux", aux)]
            if value is not None
        }

        # Extract evaluator slugs
        evaluator_slugs = None
        if evaluators:
            evaluator_slugs = []
            for evaluator in evaluators:
                if isinstance(evaluator, str):
                    evaluator_slugs.append(evaluator)
                elif isinstance(evaluator, EvaluatorDetails):
                    evaluator_slugs.append(evaluator.slug)

        # Prepare request payload
        request_body = RunInGithubRequest(
            experiment_slug=experiment_slug,
            dataset_slug=dataset_slug,
            dataset_version=dataset_version,
            evaluator_slugs=evaluator_slugs,
            task_results=task_results,
            github_context=github_context,
            experiment_metadata=experiment_metadata,
            experiment_run_metadata=experiment_run_metadata,
        )

        response = self._http_client.post(
            "/experiments/run-in-github",
            request_body.model_dump(mode="json", exclude_none=True),
        )

        if response is None:
            raise Exception(
                f"Failed to submit experiment '{experiment_slug}' for GitHub execution. "
            )

        return RunInGithubResponse(**response)

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
        task_input: Optional[Dict[str, Any]],
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

    async def _execute_tasks(
        self,
        rows: List[Dict[str, Any]],
        task: Callable[[Optional[Dict[str, Any]]], Awaitable[Dict[str, Any]]],
        evaluators: Optional[List[EvaluatorSpec]] = None,
    ) -> List[TaskResult]:
        """Execute tasks locally with concurrency control

        Args:
            rows: List of dataset rows to process
            task: Function to run on each row
            evaluators: List of evaluators to validate task output against

        Returns:
            List of TaskResult objects with inputs, outputs, and errors
        """
        task_results: List[TaskResult] = []

        # Extract EvaluatorDetails from evaluators list
        evaluators_to_validate = []
        if evaluators:
            for evaluator in evaluators:
                if isinstance(evaluator, EvaluatorDetails):
                    evaluators_to_validate.append(evaluator)

        async def run_single_row(row: Optional[Dict[str, Any]]) -> TaskResult:
            try:
                task_output = await task(row)

                # Validate task output schema and normalize field names using synonyms
                if evaluators_to_validate:
                    try:
                        task_output = validate_and_normalize_task_output(task_output, evaluators_to_validate)
                    except ValueError as validation_error:
                        print(f"\033[91m❌ Task validation failed: {str(validation_error)}\033[0m")
                        raise ValueError(str(validation_error))

                return TaskResult(
                    input=row,
                    output=task_output,
                )
            except Exception as e:
                if isinstance(e, ValueError):
                    raise e
                print(f"\033[91m❌ Task execution error: {str(e)}\033[0m")
                return TaskResult(
                    input=row,
                    error=str(e),
                )

        # Execute tasks with concurrency control
        semaphore = asyncio.Semaphore(50)

        async def run_with_semaphore(row: Dict[str, Any]) -> TaskResult:
            async with semaphore:
                return await run_single_row(row)

        tasks = [asyncio.create_task(run_with_semaphore(row)) for row in rows]

        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            task_results.append(result)

        return task_results
