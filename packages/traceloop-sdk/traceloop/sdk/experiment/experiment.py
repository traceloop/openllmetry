import uuid
import asyncio
from typing import Any, List, Callable, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.datasets.datasets import Datasets
from traceloop.sdk.evaluator.evaluator import Evaluator
from traceloop.sdk.dataset.row import Row   

class Experiment(): 
    """Main Experiment class for creating experiment contexts"""
    _datasets: Datasets
    _evaluator: Evaluator
    _http_client: HTTPClient
 
    def __init__(self, http_client: HTTPClient):
        self._datasets = Datasets(http_client)
        self._evaluator = Evaluator(http_client)
        self._http_client = http_client

    def run(
        self,
        task: Callable[[Optional[Row]], Any],
        dataset_slug: Optional[str] = None,
        evaluators: Optional[List[str]] = None,
        experiment_name: Optional[str] = None,
        concurrency: int = 10,
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
        
        if dataset_slug:
            dataset = self._datasets.get_by_slug(dataset_slug)

            
        results = []
        errors = []
        
        def run_single_row(row):
            try:
                # Run the task function
                result = task(row)
                
                # Run evaluators if provided
                eval_results = {}
                if evaluators:
                    for evaluator_slug in evaluators:
                        try:
                            eval_result = asyncio.run(self._evaluator.run(
                                evaluator_slug=evaluator_slug,
                                input={"completion": result},
                                timeout_in_sec=120
                            ))
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
        
        # Run tasks with concurrency
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(run_single_row, row) for row in dataset.rows]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
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
        
        print(f"Experiment '{experiment_name}' completed with {len(results)} successful results and {len(errors)} errors")
        
        return experiment_id, {
            "results": results,
            "errors": errors,
            "experiment_name": experiment_name,
            "experiment_id": experiment_id
        }
    
    
    