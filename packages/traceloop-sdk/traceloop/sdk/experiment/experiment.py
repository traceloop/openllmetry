from math import exp
import uuid
import asyncio
from typing import Any, List, Callable, Optional, Tuple, Dict
from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.datasets.datasets import Datasets
from traceloop.sdk.evaluator.evaluator import Evaluator
from traceloop.sdk.dataset.row import Row   
from traceloop.sdk.experiment.model import RunContextData, CreateExperimentRequest

class Experiment(): 
    """Main Experiment class for creating experiment contexts"""
    _datasets: Datasets
    _evaluator: Evaluator
    _http_client: HTTPClient
 
    def __init__(self, http_client: HTTPClient):
        self._datasets = Datasets(http_client)
        self._evaluator = Evaluator(http_client)
        self._http_client = http_client

    async def run(
        self,
        task: Callable[[Optional[Row]], Dict[str, Any]],
        dataset_slug: Optional[str] = None,
        evaluators: Optional[List[str]] = None,
        experiment_slug: Optional[str] = None,
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
        run_id = str(uuid.uuid4())

        if experiment_slug:
            experiment = self._get_experiment_by_slug(experiment_slug)
        else:
            experiment = self._create_experiment()
        
        if dataset_slug:
            dataset = self._datasets.get_by_slug(dataset_slug)

            
        results = []
        errors = []
        
        async def run_single_row(row):
            try:
                # Run the task function
                result = task(row)
                task_id = str(uuid.uuid4())
                context_data = RunContextData(
                    experiment_id=experiment_id,
                    run_id=run_id,
                    task_id=task_id,
                    task_input=row.values,
                    task_output=result,
                    dataset_ids=[dataset.id],
                    evaluator_slugs=evaluators,
                    evaluator_versions=None
                )
                
                # Run evaluators if provided
                eval_results = {}
                if evaluators:
                    for evaluator_slug in evaluators:
                        try:
                            eval_result = await self._evaluator.run(
                                evaluator_slug=evaluator_slug,
                                input={"completion": result},
                                timeout_in_sec=120,
                            )
                            eval_results[evaluator_slug] = eval_result.result
                        except Exception as e:
                            eval_results[evaluator_slug] = f"Error: {str(e)}"
                            
                return {
                    "row_id": getattr(row, 'id', None),
                    "input": row.values,
                    "output": result,
                    "evaluations": eval_results
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
        
        tasks = [run_with_semaphore(row) for row in dataset.rows]
        
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
        
        experiment_id = str(uuid.uuid4())
        
        print(f"Experiment '{experiment_slug}' completed with {len(results)} successful results and {len(errors)} errors")
        
        return experiment_id, {
            "results": results,
            "errors": errors,
            "experiment_name": experiment_slug,
            "experiment_id": experiment_id
        }
    
    def _get_experiment_by_slug(self, experiment_slug: str) -> Dict[str, Any]:
        """Get experiment by slug from API"""
        response = self._http_client.get(f"/v2/experiments/{experiment_slug}")
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Experiment with slug '{experiment_slug}' not found")
    
    def _create_experiment(self, dataset_ids: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new experiment"""
        experiment_slug = "exp-" + str(uuid.uuid4())[:11]
        body = CreateExperimentRequest(
            slug=experiment_slug,
            dataset_ids=dataset_ids,
            metadata=metadata
        )
        response = self._http_client.post("/v2/experiments", json=body.model_dump())
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception("Failed to create experiment")
    
    
    