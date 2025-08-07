import time
import asyncio
from typing import Dict, List

from traceloop.sdk.evaluators.evaluator import Evaluator
from .model import ExperimentResult, ExperimentRunResult


class Experiment:
    """
    Experiment class for running a single evaluator on multiple inputs
    """
    id: str
    name: str
    description: str

    @classmethod
    async def run(cls, 
                  evaluator_slug: str,
                  inputs: List[Dict[str, str]], 
                  timeout_in_sec: int = 120) -> ExperimentResult:
        """
        Execute experiment: run single evaluator on multiple inputs in parallel
        
        Args:
            evaluator_slug: Slug of the evaluator to execute
            inputs: List of input dictionaries for the evaluator
            timeout_in_sec: Timeout in seconds for each evaluator run
        
        Returns:
            ExperimentResult: Aggregated results from all runs
        """
        start_time = time.time()
        
        # Create shared HTTP client for all evaluator runs
        client = await Evaluator._create_async_client()
        
        try:
            tasks = [
                cls._execute_single_run(evaluator_slug, input_data, i, client, timeout_in_sec)
                for i, input_data in enumerate(inputs)
            ]
            
            # Execute all evaluator runs in parallel
            results = await asyncio.gather(*tasks, return_exceptions=False)
            
        finally:
            # Always close the client when done
            await client.aclose()
        
        total_execution_time = time.time() - start_time
        
        # Aggregate results
        successful_runs = len([r for r in results if r.error is None])
        failed_runs = len([r for r in results if r.error is not None])
        
        return ExperimentResult(
            evaluator_slug=evaluator_slug,
            total_runs=len(inputs),
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            results=results,
            total_execution_time=total_execution_time
        )


    @classmethod
    async def _execute_single_run(cls, 
                                  evaluator_slug: str, 
                                  input_data: Dict[str, str], 
                                  input_index: int,
                                  client,
                                  timeout_in_sec: int) -> ExperimentRunResult:
        """Execute a single evaluator run and capture result/error"""
        run_start = time.time()
        
        try:
            result = await Evaluator.run(evaluator_slug, input_data, client, timeout_in_sec)
            execution_time = time.time() - run_start
            
            return ExperimentRunResult(
                input_index=input_index,
                input_data=input_data,
                result=result,
                error=None,
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - run_start
            
            return ExperimentRunResult(
                input_index=input_index,
                input_data=input_data,
                result=None,
                error=str(e),
                execution_time=execution_time
            )

