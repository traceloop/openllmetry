import time
import asyncio
from typing import Dict, List, Iterator, Optional

from traceloop.sdk.evaluators.evaluator import Evaluator
from .model import ExperimentResult, ExperimentRunResult


class Experiment:
    """
    Experiment class for running a single evaluator on multiple inputs
    """
    id: str
    name: str
    description: str

    @staticmethod
    def _chunk_inputs(inputs: List[Dict[str, str]], chunk_size: int = 100) -> Iterator[List[Dict[str, str]]]:
        """Split inputs into chunks of specified size"""
        for i in range(0, len(inputs), chunk_size):
            yield inputs[i:i + chunk_size]

    @classmethod
    async def evaluate(cls,
                  evaluator_slug: str,
                  inputs: List[Dict[str, str]],
                  timeout_in_sec: int = 120,
                  dataset_slug: Optional[str] = None) -> ExperimentResult:
        """
        Execute experiment: run single evaluator on multiple inputs in parallel
        Process inputs in batches of 100, using a new HTTP client for each batch.

        Args:
            evaluator_slug: Slug of the evaluator to execute
            inputs: List of input dictionaries for the evaluator
            timeout_in_sec: Timeout in seconds for each evaluator run
            dataset_slug: Slug of the dataset to use for the experiment

        Returns:
            ExperimentResult: Aggregated results from all runs
        """
        start_time = time.time()
        all_results = []

        # Process inputs in batches of 100
        for batch_inputs in cls._chunk_inputs(inputs, chunk_size=100):
            batch_results = await cls._process_batch(
                evaluator_slug,
                batch_inputs,
                timeout_in_sec,
                start_index=len(all_results)
            )
            all_results.extend(batch_results)

        total_execution_time = time.time() - start_time

        # Aggregate results
        successful_runs = len([r for r in all_results if r.error is None])
        failed_runs = len([r for r in all_results if r.error is not None])

        return ExperimentResult(
            evaluator_slug=evaluator_slug,
            total_runs=len(inputs),
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            results=all_results,
            total_execution_time=total_execution_time
        )

    @classmethod
    async def _process_batch(cls,
                             evaluator_slug: str,
                             batch_inputs: List[Dict[str, str]],
                             timeout_in_sec: int,
                             start_index: int) -> List[ExperimentRunResult]:
        """
        Process a batch of inputs with a dedicated HTTP client

        Args:
            evaluator_slug: Slug of the evaluator to execute
            batch_inputs: List of input dictionaries for this batch
            timeout_in_sec: Timeout in seconds for each evaluator run
            start_index: Starting index for this batch (for proper indexing)

        Returns:
            List of ExperimentRunResult for this batch
        """
        # Create new HTTP client for this batch
        client = Evaluator._create_async_client()

        try:
            tasks = [
                cls._execute_single_run(
                    evaluator_slug,
                    input_data,
                    start_index + i,
                    client,
                    timeout_in_sec
                )
                for i, input_data in enumerate(batch_inputs)
            ]

            # Execute all evaluator runs in this batch in parallel
            results = await asyncio.gather(*tasks, return_exceptions=False)
            return results

        finally:
            await client.aclose()

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
            result = await Evaluator.run(evaluator_slug, input_data, timeout_in_sec, client)
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
