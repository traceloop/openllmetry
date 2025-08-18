import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from .model import EvaluatorRun, EvalRun, ExperimentData
from traceloop.sdk.evaluator.evaluator import Evaluator as BaseEvaluator
from traceloop.sdk.client.http import HTTPClient


class ExperimentContext:
    """Context manager for experiments that automatically logs results on exit"""
    
    def __init__(self, http: HTTPClient):
        self.api_client = http
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically log experiment data via POST API"""
        self.completed_at = datetime.now()
        
        # Complete any pending eval run
        if self.current_eval_run is not None:
            self._complete_current_eval_run()
        
        # Prepare and send experiment data
        experiment_data = self._prepare_experiment_data()
        
        try:
            self._post_experiment_results(experiment_data)
            print(f"✅ Experiment '{self.experiment_name}' logged successfully")
        except Exception as e:
            print(f"❌ Failed to log experiment results: {e}")
            self._save_backup_results(experiment_data)
            
        return False  # Don't suppress exceptions
    
    def start_eval_run(self, 
                       dataset_id: str, 
                       dataset_name: str, 
                       dataset_row_id: str, 
                       variant_name: str = "default"):
        """Start tracking a new evaluation run"""
        if self.current_eval_run is not None:
            self._complete_current_eval_run()
            
        self.current_eval_run = EvalRun(
            eval_run_id=str(uuid.uuid4()),
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            dataset_row_id=dataset_row_id,
            variant_name=variant_name,
            started_at=datetime.now(),
            completed_at=None,
            evaluator_runs=[]
        )
        
    def log_evaluator_run(self, 
                         evaluator_id: str, 
                         evaluator_name: str,
                         input_data: Dict[str, Any], 
                         output_data: Dict[str, Any],
                         runtime_ms: float, 
                         success: bool = True, 
                         error_message: Optional[str] = None):
        """Log individual evaluator execution"""
        if self.current_eval_run is None:
            raise ValueError("Must call start_eval_run() before logging evaluator runs")
            
        evaluator_run = EvaluatorRun(
            evaluator_id=evaluator_id,
            evaluator_name=evaluator_name,
            input_data=input_data,
            output_data=output_data,
            runtime_ms=runtime_ms,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            success=success,
            error_message=error_message
        )
        
        self.current_eval_run.evaluator_runs.append(evaluator_run)
        
    def complete_eval_run(self):
        """Mark current evaluation run as complete"""
        if self.current_eval_run is not None:
            self._complete_current_eval_run()
            
    def _complete_current_eval_run(self):
        """Internal method to complete current eval run"""
        if self.current_eval_run is not None:
            self.current_eval_run.completed_at = datetime.now()
            self.eval_runs.append(self.current_eval_run)
            self.current_eval_run = None
            
    def _prepare_experiment_data(self) -> ExperimentData:
        """Prepare experiment data for API logging"""
        total_evaluator_runs = sum(len(run.evaluator_runs) for run in self.eval_runs)
        successful_runs = sum(
            1 for run in self.eval_runs 
            for eval_run in run.evaluator_runs 
            if eval_run.success
        )
        success_rate = successful_runs / total_evaluator_runs if total_evaluator_runs > 0 else 0.0
        
        duration_ms = (self.completed_at - self.created_at).total_seconds() * 1000
        
        return ExperimentData(
            experiment_id=self.experiment_id,
            experiment_name=self.experiment_name,
            created_at=self.created_at,
            completed_at=self.completed_at,
            duration_ms=duration_ms,
            eval_runs=self.eval_runs,
            total_evaluator_runs=total_evaluator_runs,
            success_rate=success_rate,
            metadata={}
        )
        
    def _post_experiment_results(self, experiment_data: ExperimentData):
        """POST experiment results to API"""
        payload = {
            "experiment_id": experiment_data.experiment_id,
            "experiment_name": experiment_data.experiment_name,
            "created_at": experiment_data.created_at.isoformat(),
            "completed_at": experiment_data.completed_at.isoformat() if experiment_data.completed_at else None,
            "duration_ms": experiment_data.duration_ms,
            "total_evaluator_runs": experiment_data.total_evaluator_runs,
            "success_rate": experiment_data.success_rate,
            "eval_runs": [
                {
                    "eval_run_id": run.eval_run_id,
                    "dataset_id": run.dataset_id,
                    "dataset_name": run.dataset_name,
                    "dataset_row_id": run.dataset_row_id,
                    "variant_name": run.variant_name,
                    "started_at": run.started_at.isoformat(),
                    "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                    "evaluator_runs": [
                        {
                            "evaluator_id": eval_run.evaluator_id,
                            "evaluator_name": eval_run.evaluator_name,
                            "input_data": eval_run.input_data,
                            "output_data": eval_run.output_data,
                            "runtime_ms": eval_run.runtime_ms,
                            "started_at": eval_run.started_at.isoformat(),
                            "completed_at": eval_run.completed_at.isoformat(),
                            "success": eval_run.success,
                            "error_message": eval_run.error_message
                        }
                        for eval_run in run.evaluator_runs
                    ]
                }
                for run in experiment_data.eval_runs
            ]
        }
        
        response = self.api_client.post(
            "/experiments/results", 
            json=payload
        )
        response.raise_for_status()
        
    def _save_backup_results(self, experiment_data: ExperimentData):
        """Save results to local file if API fails"""
        backup_file = f"experiment_{self.experiment_id}_{int(time.time())}.json"
        
        # Convert dataclass to dict for JSON serialization
        backup_data = {
            "experiment_id": experiment_data.experiment_id,
            "experiment_name": experiment_data.experiment_name,
            "created_at": experiment_data.created_at.isoformat(),
            "completed_at": experiment_data.completed_at.isoformat() if experiment_data.completed_at else None,
            "duration_ms": experiment_data.duration_ms,
            "total_evaluator_runs": experiment_data.total_evaluator_runs,
            "success_rate": experiment_data.success_rate,
            "eval_runs": [
                {
                    "eval_run_id": run.eval_run_id,
                    "dataset_id": run.dataset_id,
                    "dataset_name": run.dataset_name,
                    "dataset_row_id": run.dataset_row_id,
                    "variant_name": run.variant_name,
                    "started_at": run.started_at.isoformat(),
                    "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                    "evaluator_runs": [
                        {
                            "evaluator_id": eval_run.evaluator_id,
                            "evaluator_name": eval_run.evaluator_name,
                            "input_data": eval_run.input_data,
                            "output_data": eval_run.output_data,
                            "runtime_ms": eval_run.runtime_ms,
                            "started_at": eval_run.started_at.isoformat(),
                            "completed_at": eval_run.completed_at.isoformat(),
                            "success": eval_run.success,
                            "error_message": eval_run.error_message
                        }
                        for eval_run in run.evaluator_runs
                    ]
                }
                for run in experiment_data.eval_runs
            ]
        }
        
        try:
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)
            print(f"💾 Results saved to backup file: {backup_file}")
        except Exception as e:
            print(f"❌ Failed to save backup: {e}")


class TrackedEvaluator:
    """Wrapper around evaluator to automatically track execution"""
    
    def __init__(self, evaluator_slug: str, experiment_context: ExperimentContext):
        self.evaluator_slug = evaluator_slug
        self.experiment_context = experiment_context
        
    async def evaluate(self, **kwargs) -> Any:
        """Execute evaluator with automatic tracking"""
        start_time = time.perf_counter()
        
        try:
            # Execute the actual evaluator
            result = await BaseEvaluator.run(
                evaluator_slug=self.evaluator_slug,
                input=kwargs
            )
            
            # Calculate runtime
            end_time = time.perf_counter()
            runtime_ms = (end_time - start_time) * 1000
            
            # Log successful run
            self.experiment_context.log_evaluator_run(
                evaluator_id=self.evaluator_slug,
                evaluator_name=self.evaluator_slug,  # Use slug as name for now
                input_data=kwargs,
                output_data=result if isinstance(result, dict) else {"result": result},
                runtime_ms=runtime_ms,
                success=True
            )
            
            return result
            
        except Exception as e:
            # Calculate runtime even for failures
            end_time = time.perf_counter()
            runtime_ms = (end_time - start_time) * 1000
            
            # Log failed run
            self.experiment_context.log_evaluator_run(
                evaluator_id=self.evaluator_slug,
                evaluator_name=self.evaluator_slug,
                input_data=kwargs,
                output_data={},
                runtime_ms=runtime_ms,
                success=False,
                error_message=str(e)
            )
            
            # Re-raise the exception
            raise e


class Experiment:
    """Main Experiment class for creating experiment contexts"""
    
    @staticmethod
    def create(name: str) -> ExperimentContext:
        """Create a new experiment context"""
        return ExperimentContext(name)
    
    @staticmethod
    def get_evaluator(evaluator_slug: str, experiment_context: ExperimentContext) -> TrackedEvaluator:
        """Get evaluator with automatic tracking"""
        return TrackedEvaluator(evaluator_slug, experiment_context)