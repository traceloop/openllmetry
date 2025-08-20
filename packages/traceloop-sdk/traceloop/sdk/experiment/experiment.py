from re import S
import cuid
import asyncio
from typing import Any, List, Callable, Optional, Tuple, Dict
from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.datasets.datasets import Datasets
from traceloop.sdk.evaluator.evaluator import Evaluator
from traceloop.sdk.dataset.row import Row
from traceloop.sdk.experiment.model import (
    RunContextData,
    InitExperimentRequest,
    ExperimentInitResponse,
)


class Experiment:
    """Main Experiment class for creating experiment contexts"""

    _datasets: Datasets
    _evaluator: Evaluator
    _http_client: HTTPClient

    def __init__(self, http_client: HTTPClient):
        self._datasets = Datasets(http_client)
        self._evaluator = Evaluator()
        self._http_client = http_client

    async def run(
        self,
        task: Callable[[Optional[Row]], Dict[str, Any]],
        dataset_slug: Optional[str] = None,
        evaluators: Optional[List[str]] = None,
        experiment_slug: Optional[str] = None,
        related_ref: Optional[Dict[str, str]] = None,
        aux: Optional[Dict[str, str]] = None,
        exit_on_error: bool = False,
    ) -> Tuple[str, Any]:
        """Run an experiment with the given task and evaluators

        Args:
            dataset_slug: Slug of the dataset to use
            task: Function to run on each dataset row
            evaluators: List of evaluator slugs to run
            experiment_name: Name for this experiment run
            concurrency: Number of concurrent tasks
            exit_on_error: Whether to exit on first error
            client: Traceloop client instance (if not provided, will initialize)

        Returns:
            Tuple of (experiment_id, results)
        """

        if not experiment_slug:
            experiment_slug = "exp-" + str(cuid.cuid())[:11]

        experiment_metadata = {
            key: value for key, value in [
                ("related_ref", related_ref),
                ("aux", aux)
            ] if value is not None
        }

        experiment = self._init_experiment(
            experiment_slug,
            dataset_slugs=[dataset_slug],
            evaluator_slugs=evaluators,
            experiment_metadata=experiment_metadata,
        )

        run_id = experiment.run.id
        print(f"AASA = Run ID: {run_id}")

        if dataset_slug:
            dataset = self._datasets.get_by_slug(dataset_slug)

        results = []
        errors = []

        async def run_single_row(row):
            try:
                # Run the task function
                task_result = task(row)
                print(f"AASA = Result: {task_result}")
                task_id = str(cuid.cuid())

                # Run evaluators if provided
                eval_results = {}
                print(f"AASA = Evaluators: {evaluators}")
                if evaluators:
                    for evaluator_slug in evaluators:
                        try:
                            context_data = RunContextData(
                                experiment_id=experiment.id,
                                experiment_run_id=run_id,
                                task_id=task_id,
                                task_input=row.values,
                                task_output=task_result,
                                dataset_slugs=[dataset_slug] if dataset_slug else [],
                                evaluator_slug=evaluator_slug,
                                evaluator_version=None,
                            )

                            print(f"AASA = Evaluator slug: {evaluator_slug}")
                            eval_result = await self._evaluator.run(
                                evaluator_slug=evaluator_slug,
                                input=task_result,
                                timeout_in_sec=120,
                                context_data=context_data.model_dump(),
                            )
                            print(f"AASA = Evaluator result: {eval_result}")
                            eval_results[evaluator_slug] = eval_result.result
                        except Exception as e:
                            eval_results[evaluator_slug] = f"Error: {str(e)}"

                return {
                    "row_id": getattr(row, "id", None),
                    "input": row.values,
                    "output": task_result,
                    "evaluations": eval_results,
                }
            except Exception as e:
                error_msg = f"Error processing row: {str(e)}"
                if exit_on_error:
                    raise e
                return {"error": error_msg}

        semaphore = asyncio.Semaphore(50)

        async def run_with_semaphore(row):
            async with semaphore:
                return await run_single_row(row)

        tasks = [
            run_with_semaphore(row) for row in dataset.rows[:1]
        ]  # Only 1 task for debug

        for completed_task in asyncio.as_completed(tasks):
            try:
                result = await completed_task
                if "error" in result:
                    errors.append(result["error"])
                else:
                    results.append(result)
            except Exception as e:
                error_msg = f"Task execution error: {str(e)}"
                errors.append(error_msg)
                if exit_on_error:
                    break

        print(
            f"Experiment '{experiment_slug}' completed with {len(results)} successful results and {len(errors)} errors"
        )

        print("\n\nERRORS: ", errors)

        return experiment.id, {
            "results": results,
            "errors": errors,
            "experiment_name": experiment_slug,
            "experiment_id": experiment.id,
        }

    def _init_experiment(
        self,
        experiment_slug: str,
        dataset_slugs: Optional[List[str]] = None,
        evaluator_slugs: Optional[List[str]] = None,
        experiment_metadata: Optional[Dict[str, Any]] = None,
        experiment_run_metadata: Optional[Dict[str, Any]] = None,
    ) -> ExperimentInitResponse:
        """Get experiment by slug from API"""
        body = InitExperimentRequest(
            slug=experiment_slug,
            dataset_slugs=dataset_slugs,
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
