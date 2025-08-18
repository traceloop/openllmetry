import json
import os
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from .model import EvaluatorRun, EvalRun, ExperimentData
from traceloop.sdk.evaluator.evaluator import Evaluator as BaseEvaluator
from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.version import __version__


class ExperimentContext:
    """Context manager for experiments that automatically logs results on exit"""
    
    def __init__(self, experiment: 'Experiment'):
        self.experiment = experiment
        
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
            
        return False
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
        

class Experiment:
    """Main Experiment class for creating experiment contexts"""
    id: str
    name: str
    created_at: datetime
    run_data: Dict[str, Any]
 
    def __init__(self, name: str):
        self.name = name
        self.created_at = datetime.now()
        self.id = str(uuid.uuid4())
        self._http_client = self._get_http_client()

    def _get_http_client(self) -> HTTPClient:
        api_key = os.getenv("TRACELOOP_API_KEY")
        if not api_key:
            raise Exception("TRACELOOP_API_KEY is not set")
        api_endpoint = os.getenv("TRACELOOP_BASE_URL", "https://api.traceloop.com")
        return HTTPClient(
            base_url=api_endpoint, api_key=api_key, version=__version__
        )
    
    def run(self, http: HTTPClient = None) -> ExperimentContext:
        """Create a new experiment context"""
        return ExperimentContext(self, http or self._http_client)
    